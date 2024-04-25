#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>
#include <cmath>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val; 
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors
        //loop over Batch Size
         for (int b = 0; b < B; b++) {
             //loop over Heads
             for (int h = 0; h < H; h++) {
                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {
                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    
    //loop over Batch Size
    for (int b = 0; b < B; b++) {
        //loop over Heads
        for (int h = 0; h < H; h++) {
            //loop over Sequence Length
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0;
                    //loop over Embedding Dimensionality
                    for (int k = 0; k < d; k++) {
                        float q_val = fourDimRead(Q, b, h, i, k, H, N, d);
                        float kt_val = fourDimRead(K, b, h, j, k, H, N, d);
                        sum += q_val * kt_val;
                    }
                    twoDimWrite(QK_t, i, j, N, sum); // N x N
                }
            }

            // Apply softmax
            for (int i = 0; i < N; i++) {
                float sum = 0.0;
                for (int j = 0; j < N; j++) {
                    float exp_val = exp(twoDimRead(QK_t, i, j, N));
                    sum += exp_val;
                    twoDimWrite(QK_t, i, j, N, exp_val);
                }
                for (int j = 0; j < N; j++) {
                    float norm_val = twoDimRead(QK_t, i, j, N) / sum;
                    twoDimWrite(QK_t, i, j, N, norm_val);
                }
            }

            // Matrix multiply QK^t * V
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < d; j++) {
                    float sum = 0.0;
                    for (int k = 0; k < N; k++) {
                        float v_val = fourDimRead(V, b, h, k, j, H, N, d);
                        float qkt_val = twoDimRead(QK_t, i, k, N);
                        sum += v_val * qkt_val;
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, sum);
                }
            }
        }
    }
    
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    int BLOCK_SIZE_N = 8, BLOCK_SIZE_D = 8;

    //loop over Batch Size
    float zero = 0.0;
    for (int b = 0; b < B; b++) {
        //loop over Heads
        for (int h = 0; h < H; h++) {
            // Set QK_t to zeros
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    twoDimWrite(QK_t, i, j, N, zero);
                }
            }

            // Matrix multiply QK^t
            for (int iblock = 0; iblock < N; iblock += BLOCK_SIZE_N) {
                for (int jblock = 0; jblock < N; jblock += BLOCK_SIZE_N) {
                    for (int kblock = 0; kblock < d; kblock += BLOCK_SIZE_D) {
                        //loop over Sequence Length
                        for (int i = 0; i < std::min(BLOCK_SIZE_N, N-iblock); i++) {
                            for (int j = 0; j < std::min(BLOCK_SIZE_N, N-jblock); j++) {
                                int i_index = i + iblock;
                                int j_index = j + jblock;
                                float sum = twoDimRead(QK_t, i_index, j_index, N);
                                //loop over Embedding Dimensionality
                                for (int k = 0; k < std::min(BLOCK_SIZE_D, d-kblock); k++) {
                                    int k_index = k + kblock;
                                    float q_val = fourDimRead(Q, b, h, i_index, k_index, H, N, d);
                                    float kt_val = fourDimRead(K, b, h, j_index, k_index, H, N, d);
                                    sum += q_val * kt_val;
                                }
                                twoDimWrite(QK_t, i_index, j_index, N, sum); // N x N
                            }
                        }
                    }
                }
            }

            // Q: ((N/BLOCK_SIZE_N) * BLOCK_SIZE_D)/64
            // K^T: ((N/BLOCK_SIZE_N) * BLOCK_SIZE_D)/64

            number_of_accesses_Q_blocks * number_of_accesses_KT_blocks
            = (number of Q blocks) * (number of access per Q block) * (number of KT blocks) * (number of access per KT block)
            = (N/BLOCK_SIZE_N * d/BLOCK_SIZ_D) * (BLOCK_SIZE_N * BLOCK_SIZE_D / 64) * (N/BLOCK_SIZE_N * d/BLOCK_SIZ_D) * (BLOCK_SIZE_N * BLOCK_SIZE_D / 64)
            = (N/BLOCK_SIZE_N * d/BLOCK_SIZ_D)^2 * (BLOCK_SIZE_N * BLOCK_SIZE_D / 64)^2


            number_of_accesses_Q_blocks + (number of Q blocks) * number_of_accesses_KT_blocks
            = (number of Q blocks) * ((number of access per Q block) + (number of KT blocks) * (number of access per KT block))
            = (N/BLOCK_SIZE_N * d/BLOCK_SIZ_D) * ((BLOCK_SIZE_N * BLOCK_SIZE_D / 64) + (N/BLOCK_SIZE_N * d/BLOCK_SIZ_D) * (BLOCK_SIZE_N * BLOCK_SIZE_D / 64))
            = (N/BLOCK_SIZE_N * d/BLOCK_SIZ_D) * ((BLOCK_SIZE_N * BLOCK_SIZE_D / 64)(1 + (N/BLOCK_SIZE_N * d/BLOCK_SIZ_D)))
            = (N/8 * d/8) * ((8 * 8 / 64)(1 + (N/8 * d/8)))
            = (N/8 * d/8) * ((1)(1 + (N/8 * d/8)))
            = (N/8 * d/8) * (1 + (N/8 * d/8))

            // std::cout << "After loop" << std::endl;
            // if (b == 0 && h == 0) {
            //     std::vector<int> vect = {0, 100, 1023};
            //     for (auto i : vect) {
            //         std::cout << "ind i = [";
            //         for (int j = 0; j < 15; j++) {
            //             std::cout << twoDimRead(QK_t, i, j, N) << ", ";
            //         }
            //         std::cout << "]" << std::endl;
            //     }
            // }
            // Apply softmax
            for (int i = 0; i < N; i++) {
                float sum = 0.0;
                for (int j = 0; j < N; j++) {
                    float exp_val = exp(twoDimRead(QK_t, i, j, N));
                    sum += exp_val;
                    twoDimWrite(QK_t, i, j, N, exp_val);
                }
                for (int j = 0; j < N; j++) {
                    float norm_val = twoDimRead(QK_t, i, j, N) / sum;
                    twoDimWrite(QK_t, i, j, N, norm_val);
                }
            }

            // Matrix multiply QK^t * V
            for (int iblock = 0; iblock < N; iblock += BLOCK_SIZE_N) {
                for (int jblock = 0; jblock < d; jblock += BLOCK_SIZE_D) {
                    for (int kblock = 0; kblock < N; kblock += BLOCK_SIZE_N) {
                        for (int i = 0; i < std::min(BLOCK_SIZE_N, N-iblock); i++) {
                            for (int j = 0; j < std::min(BLOCK_SIZE_D, d-jblock); j++) {
                                int i_index = i + iblock;
                                int j_index = j + jblock;
                                float sum = fourDimRead(O, b, h, i_index, j_index, H, N, d);
                                for (int k = 0; k < std::min(BLOCK_SIZE_N, N-kblock); k++) {
                                    int k_index = k + kblock;
                                    float v_val = fourDimRead(V, b, h, k_index, j_index, H, N, d);
                                    float qkt_val = twoDimRead(QK_t, i_index, k_index, N);
                                    sum += v_val * qkt_val;
                                }
                                fourDimWrite(O, b, h, i_index, j_index, H, N, d, sum);
                            }
                        }
                    }
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++){
        //loop over heads
        for (int h = 0; h < H; h++){
            for (int i = 0; i < N ; i++){
		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
		//YOUR CODE HERE
                for (int j = 0; j < N; j++) {
                    float sum = 0.0;
                    //loop over Embedding Dimensionality
                    for (int k = 0; k < d; k++) {
                        float q_val = fourDimRead(Q, b, h, i, k, H, N, d);
                        float kt_val = fourDimRead(K, b, h, j, k, H, N, d);
                        sum += q_val * kt_val;
                    }
                    ORow[j] = sum;
                }

                float sum = 0.0;
                for (int j = 0; j < N; j++) {
                    float exp_val = exp(ORow[j]);
                    sum += exp_val;
                    ORow[j] = exp_val;
                }
                for (int j = 0; j < N; j++) {
                    ORow[j] /= sum;
                }

                // Matrix multiply QK^t * V
                for (int j = 0; j < d; j++) {
                    float sum = 0.0;
                    for (int k = 0; k < N; k++) {
                        float v_val = fourDimRead(V, b, h, k, j, H, N, d);
                        float qkt_val = ORow[k];
                        sum += v_val * qkt_val;
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, sum);
                }
            }
	    }
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    int Tr = ceil(N / Br);
    int Tc = ceil(N / Bc);

    // std::cout << "Br: " << Br << ", Bc: " << Bc << std::endl; // 256, 256
    // std::cout << "Tr: " << Tr << ", Tc: " << Tc << std::endl; // 4, 4

    for (int b = 0; b < B; b++){
        //loop over heads
        for (int h = 0; h < H; h++){
            std::fill(l.begin(), l.end(), 0);
            for (int j = 0; j < N; j += Bc) {
                // load Kj and Vj into memory (Bc, d)
                // std::cout << "load Kj and Vj into memory (Bc, d)" << std::endl;
                for (int jj = 0; jj < std::min(Bc, N-j); jj++) {
                    int j_index = j + jj;
                    for (int k = 0; k < d; k++) {
                        float kj_val = fourDimRead(K, b, h, j_index, k, H, N, d);
                        float vj_val = fourDimRead(V, b, h, j_index, k, H, N, d);
                        twoDimWrite(Kj, jj, k, d, kj_val);
                        twoDimWrite(Vj, jj, k, d, vj_val);
                    }
                }
                // std::cout << "load Qi and Oi into memory (Br, d)" << std::endl;
                // load Qi, Oi, and li into memory (Br, d) and (Br)
                for (int i = 0; i < N; i += Br) {
                    for (int ii = 0; ii < std::min(Br, N-i); ii++) {
                        int i_index = i + ii;
                        // load Qi and Oi
                        for (int k = 0; k < d; k++) {
                            float qi_val = fourDimRead(Q, b, h, i_index, k, H, N, d);
                            twoDimWrite(Qi, ii, k, d, qi_val);

                            float oi_val = fourDimRead(O, b, h, i_index, k, H, N, d);
                            twoDimWrite(Oi, ii, k, d, oi_val);
                        }
                        // load li
                        li[ii] = l[i_index];

                        // std::cout << "compute Sij = QiKj^T (Br, Bc)" << std::endl;
                        // compute Sij = QiKj^T (Br, Bc)
                        for (int jj = 0; jj < std::min(Bc, N-j); jj++) {
                            int j_index = j + jj;
                            float sum = 0.0;
                            for (int k = 0; k < d; k++) {
                                // std::cout << "ii: " << ii << ", jj: " << jj << ", k: " << k << std::endl;
                                float qi_val = twoDimRead(Qi, ii, k, d);
                                float kj_val = twoDimRead(Kj, jj, k, d);
                                sum += qi_val * kj_val;
                            }
                            // std::cout << "ii: " << ii << ", jj: " << jj << std::endl;
                            twoDimWrite(Sij, ii, jj, Bc, sum);
                        }

                        float sum = 0.0;
                        // std::cout << "compute Pij = exp(Sij) (Br, Bc)" << std::endl;
                        // need tocheck bounds for Br and Bc
                        for (int jj = 0; jj < std::min(Bc, N-j); jj++) {
                            int j_index = j + jj;
                            float exp_val = exp(twoDimRead(Sij, ii, jj, Bc));
                            sum += exp_val;
                            twoDimWrite(Pij, ii, jj, Bc, exp_val);
                        }
                        // std::cout << "compute lij = rowsum(Pij) (Br)" << std::endl;
                        // compute lij = rowsum(Pij) (Br)
                        lij[ii] = sum;
                        // compute lnew = li + lij (Br)
                        lnew[ii] = lij[ii] + li[ii];
                        // std::cout << "compute Oi = (liOi + PijVj)/lnew (Br, d)" << std::endl;
                        // compute Oi = (liOi + PijVj)/lnew (Br, d)
                        for (int k = 0; k < d; k++) {
                            float oi_val = li[ii] * twoDimRead(Oi, ii, k, d);
                            float mm_val = 0;
                            for (int jj = 0; jj < std::min(Bc, N-j); jj++) {
                                int j_index = j + jj;
                                float pij_val = twoDimRead(Pij, ii, jj, Bc);
                                float vj_val = twoDimRead(Vj, jj, k, d);
                                mm_val += pij_val * vj_val;
                            }
                            float updated_oi_val = (oi_val + mm_val) / lnew[ii];
                            twoDimWrite(Oi, ii, k, d, updated_oi_val);
                        }
                        // std::cout << "write Oi and lnew back to O (B, H, N, d) and l (N) in memory" << std::endl;
                        // write blocks Oi and lnew back to O (B, H, N, d) and l (N) in memory
                        for (int k = 0; k < d; k++) {
                            float oi_val = twoDimRead(Oi, ii, k, d);
                            fourDimWrite(O, b, h, i_index, k, H, N, d, oi_val);
                        }
                        l[i_index] = lnew[ii];
                    }
                }
            }
        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
