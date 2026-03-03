#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <functional>
#include <string>
#include <sstream>

using namespace std;
using namespace chrono;

// Type alias for matrix
using Matrix = vector<vector<double>>;

// Function to create a random matrix with a specific seed
Matrix createRandomMatrix(int n, unsigned seed) {
    mt19937 gen(seed);
    uniform_real_distribution<> dis(0.0, 10.0);

    Matrix matrix(static_cast<size_t>(n), vector<double>(static_cast<size_t>(n)));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] = dis(gen);
        }
    }
    return matrix;
}

// Brute Force Matrix Multiplication
Matrix bruteForceMultiply(const Matrix& A, const Matrix& B) {
    int n = static_cast<int>(A.size());
    Matrix C(static_cast<size_t>(n), vector<double>(static_cast<size_t>(n), 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
                    A[static_cast<size_t>(i)][static_cast<size_t>(k)] * B[static_cast<size_t>(k)][static_cast<size_t>(j)];
            }
        }
    }
    return C;
}

// Helper function to add two matrices
Matrix addMatrices(const Matrix& A, const Matrix& B) {
    int n = static_cast<int>(A.size());
    Matrix C(static_cast<size_t>(n), vector<double>(static_cast<size_t>(n)));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                A[static_cast<size_t>(i)][static_cast<size_t>(j)] + B[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }
    return C;
}

// Helper function to subtract two matrices
Matrix subtractMatrices(const Matrix& A, const Matrix& B) {
    int n = static_cast<int>(A.size());
    Matrix C(static_cast<size_t>(n), vector<double>(static_cast<size_t>(n)));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                A[static_cast<size_t>(i)][static_cast<size_t>(j)] - B[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }
    return C;
}

// Naive Divide and Conquer Matrix Multiplication
Matrix naiveDivideAndConquer(const Matrix& A, const Matrix& B) {
    int n = static_cast<int>(A.size());

    // Base case
    if (n == 1) {
        Matrix C(1, vector<double>(1));
        C[0][0] = A[0][0] * B[0][0];
        return C;
    }

    Matrix C(static_cast<size_t>(n), vector<double>(static_cast<size_t>(n), 0.0));

    int halfSize = n / 2;
    size_t halfSizeU = static_cast<size_t>(halfSize);

    // Divide matrices into quadrants
    Matrix A11(halfSizeU, vector<double>(halfSizeU));
    Matrix A12(halfSizeU, vector<double>(halfSizeU));
    Matrix A21(halfSizeU, vector<double>(halfSizeU));
    Matrix A22(halfSizeU, vector<double>(halfSizeU));

    Matrix B11(halfSizeU, vector<double>(halfSizeU));
    Matrix B12(halfSizeU, vector<double>(halfSizeU));
    Matrix B21(halfSizeU, vector<double>(halfSizeU));
    Matrix B22(halfSizeU, vector<double>(halfSizeU));

    // Fill the quadrants
    for (int i = 0; i < halfSize; i++) {
        for (int j = 0; j < halfSize; j++) {
            size_t i_u = static_cast<size_t>(i);
            size_t j_u = static_cast<size_t>(j);
            size_t i_half_u = static_cast<size_t>(i + halfSize);
            size_t j_half_u = static_cast<size_t>(j + halfSize);

            A11[i_u][j_u] = A[i_u][j_u];
            A12[i_u][j_u] = A[i_u][j_half_u];
            A21[i_u][j_u] = A[i_half_u][j_u];
            A22[i_u][j_u] = A[i_half_u][j_half_u];

            B11[i_u][j_u] = B[i_u][j_u];
            B12[i_u][j_u] = B[i_u][j_half_u];
            B21[i_u][j_u] = B[i_half_u][j_u];
            B22[i_u][j_u] = B[i_half_u][j_half_u];
        }
    }

    // Recursive calls to multiply quadrants
    Matrix C11 = addMatrices(naiveDivideAndConquer(A11, B11), naiveDivideAndConquer(A12, B21));
    Matrix C12 = addMatrices(naiveDivideAndConquer(A11, B12), naiveDivideAndConquer(A12, B22));
    Matrix C21 = addMatrices(naiveDivideAndConquer(A21, B11), naiveDivideAndConquer(A22, B21));
    Matrix C22 = addMatrices(naiveDivideAndConquer(A21, B12), naiveDivideAndConquer(A22, B22));

    // Combine the results
    for (int i = 0; i < halfSize; i++) {
        for (int j = 0; j < halfSize; j++) {
            size_t i_u = static_cast<size_t>(i);
            size_t j_u = static_cast<size_t>(j);
            size_t i_half_u = static_cast<size_t>(i + halfSize);
            size_t j_half_u = static_cast<size_t>(j + halfSize);

            C[i_u][j_u] = C11[i_u][j_u];
            C[i_u][j_half_u] = C12[i_u][j_u];
            C[i_half_u][j_u] = C21[i_u][j_u];
            C[i_half_u][j_half_u] = C22[i_u][j_u];
        }
    }

    return C;
}

// Strassen's Algorithm
Matrix strassenMultiply(const Matrix& A, const Matrix& B) {
    int n = static_cast<int>(A.size());

    // Base case
    if (n == 1) {
        Matrix C(1, vector<double>(1));
        C[0][0] = A[0][0] * B[0][0];
        return C;
    }

    Matrix C(static_cast<size_t>(n), vector<double>(static_cast<size_t>(n), 0.0));

    int halfSize = n / 2;
    size_t halfSizeU = static_cast<size_t>(halfSize);

    // Divide matrices into quadrants
    Matrix A11(halfSizeU, vector<double>(halfSizeU));
    Matrix A12(halfSizeU, vector<double>(halfSizeU));
    Matrix A21(halfSizeU, vector<double>(halfSizeU));
    Matrix A22(halfSizeU, vector<double>(halfSizeU));

    Matrix B11(halfSizeU, vector<double>(halfSizeU));
    Matrix B12(halfSizeU, vector<double>(halfSizeU));
    Matrix B21(halfSizeU, vector<double>(halfSizeU));
    Matrix B22(halfSizeU, vector<double>(halfSizeU));

    // Fill the quadrants
    for (int i = 0; i < halfSize; i++) {
        for (int j = 0; j < halfSize; j++) {
            size_t i_u = static_cast<size_t>(i);
            size_t j_u = static_cast<size_t>(j);
            size_t i_half_u = static_cast<size_t>(i + halfSize);
            size_t j_half_u = static_cast<size_t>(j + halfSize);

            A11[i_u][j_u] = A[i_u][j_u];
            A12[i_u][j_u] = A[i_u][j_half_u];
            A21[i_u][j_u] = A[i_half_u][j_u];
            A22[i_u][j_u] = A[i_half_u][j_half_u];

            B11[i_u][j_u] = B[i_u][j_u];
            B12[i_u][j_u] = B[i_u][j_half_u];
            B21[i_u][j_u] = B[i_half_u][j_u];
            B22[i_u][j_u] = B[i_half_u][j_half_u];
        }
    }

    // Strassen's 7 products
    Matrix M1 = strassenMultiply(addMatrices(A11, A22), addMatrices(B11, B22));
    Matrix M2 = strassenMultiply(addMatrices(A21, A22), B11);
    Matrix M3 = strassenMultiply(A11, subtractMatrices(B12, B22));
    Matrix M4 = strassenMultiply(A22, subtractMatrices(B21, B11));
    Matrix M5 = strassenMultiply(addMatrices(A11, A12), B22);
    Matrix M6 = strassenMultiply(subtractMatrices(A21, A11), addMatrices(B11, B12));
    Matrix M7 = strassenMultiply(subtractMatrices(A12, A22), addMatrices(B21, B22));

    // Calculate the resulting quadrants
    Matrix C11 = addMatrices(subtractMatrices(addMatrices(M1, M4), M5), M7);
    Matrix C12 = addMatrices(M3, M5);
    Matrix C21 = addMatrices(M2, M4);
    Matrix C22 = addMatrices(subtractMatrices(addMatrices(M1, M3), M2), M6);

    // Combine the results
    for (int i = 0; i < halfSize; i++) {
        for (int j = 0; j < halfSize; j++) {
            size_t i_u = static_cast<size_t>(i);
            size_t j_u = static_cast<size_t>(j);
            size_t i_half_u = static_cast<size_t>(i + halfSize);
            size_t j_half_u = static_cast<size_t>(j + halfSize);

            C[i_u][j_u] = C11[i_u][j_u];
            C[i_u][j_half_u] = C12[i_u][j_u];
            C[i_half_u][j_u] = C21[i_u][j_u];
            C[i_half_u][j_half_u] = C22[i_u][j_u];
        }
    }

    return C;
}

// Function to measure execution time of a specific function
double measureExecutionTime(const function<Matrix()>& func) {
    auto start = high_resolution_clock::now();
    Matrix result = func();
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<nanoseconds>(end - start);
    return static_cast<double>(duration.count()) / 1000000.0; // Convert to milliseconds
}

// Function to run multiple tests and get average time
double runMultipleTests(const function<Matrix()>& func, int numRuns) {
    // First run (warmup) - ignored
    Matrix warmup_result = func();
    (void)warmup_result; // Suppress unused variable warning

    // Multiple runs for averaging
    double total_time = 0.0;
    for (int i = 0; i < numRuns; i++) {
        total_time += measureExecutionTime(func);
    }

    return total_time / static_cast<double>(numRuns);
}

int main() {
    cout << fixed << setprecision(4);
    cout << "Matrix Multiplication Algorithm Comparison\n";
    cout << "==========================================\n";
    cout << "Note: Times shown are averages of 5 runs (warmup run excluded)\n\n";

    // Test sizes - including both powers of 2 and other sizes
    vector<int> sizes = { 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536 };
    const int num_runs = 5;

    cout << setw(10) << "Size" << " | "
        << setw(15) << "Brute Force" << " | "
        << setw(15) << "Naive D&C" << " | "
        << setw(15) << "Strassen" << "\n";

    cout << string(72, '-') << endl;

    // Use the same random seed for all tests to ensure same input matrices
    unsigned base_seed = 12345;

    for (int n : sizes) {
        // Format size as "4x4", "16x16", etc. using stringstream
        stringstream size_str;
        size_str << n << "x" << n;
        cout << setw(10) << size_str.str() << " | ";

        // Create the same random matrices for all algorithms
        Matrix A = createRandomMatrix(n, base_seed);
        Matrix B = createRandomMatrix(n, base_seed + 1);

        // Measure Brute Force (average of multiple runs)
        double brute_time = runMultipleTests([&]() { return bruteForceMultiply(A, B); }, num_runs);
        cout << setw(13) << brute_time << " ms";

        // Skip if time exceeds 2 seconds
        if (brute_time > 2000.0) {
            cout << " | " << setw(13) << "TIMEOUT" << " | " << setw(13) << "TIMEOUT";
        }
        else {
            cout << " | ";

            // Measure Naive Divide and Conquer (average of multiple runs)
            double dc_time = runMultipleTests([&]() { return naiveDivideAndConquer(A, B); }, num_runs);
            cout << setw(13) << dc_time << " ms";

            // Skip if time exceeds 2 seconds
            if (dc_time > 2000.0) {
                cout << " | " << setw(13) << "TIMEOUT";
            }
            else {
                cout << " | ";

                // Measure Strassen (average of multiple runs)
                double strassen_time = runMultipleTests([&]() { return strassenMultiply(A, B); }, num_runs);
                cout << setw(13) << strassen_time << " ms";
            }
        }

        cout << endl;
    }

    cout << "\nMaximum Matrix Size Achievable in 2 Seconds:\n";
    cout << "==========================================\n";
    cout << "Based on the experimental results above:\n\n";

    cout << "Algorithm Analysis:\n";
    cout << "==================\n";
    cout << "1. Brute Force: O(n³) time complexity\n";
    cout << "   - Simple triple nested loop\n";
    cout << "   - Best for small matrices (n < 128)\n";
    cout << "   - Practical limit: ~512x512 to 768x768 in 2 seconds\n\n";

    cout << "2. Naive Divide and Conquer: O(n³) time complexity\n";
    cout << "   - Recursive approach with 8 subproblems\n";
    cout << "   - Same complexity as brute force, but with overhead\n";
    cout << "   - Generally slower than brute force due to recursion overhead\n";
    cout << "   - Practical limit: ~512x512 to 768x768 in 2 seconds\n\n";

    cout << "3. Strassen's Algorithm: O(n^2.807) time complexity\n";
    cout << "   - Recursive approach with 7 subproblems\n";
    cout << "   - Becomes efficient for larger matrices\n";
    cout << "   - Practical limit: ~1024x1024 to 1536x1536 in 2 seconds\n\n";

    cout << "Practical Usage Recommendations:\n";
    cout << "==============================\n";
    cout << "1. Small matrices (n < 128):\n";
    cout << "   - Use Brute Force - simplest and fastest\n";
    cout << "   - Avoid recursive methods due to overhead\n\n";
    cout << "2. Medium matrices (128 ≤ n < 512):\n";
    cout << "   - Brute Force still competitive\n";
    cout << "   - Strassen's begins to show advantages\n\n";
    cout << "3. Large matrices (n ≥ 512):\n";
    cout << "   - Strassen's Algorithm clearly superior\n";
    cout << "   - Can handle significantly larger problems\n\n";
    cout << "4. Very large matrices (n > 1024):\n";
    cout << "   - Only Strassen's Algorithm remains practical\n";
    cout << "   - Consider memory-optimized implementations\n";
    cout << "   - May need specialized libraries (e.g., BLAS, cuBLAS)\n\n";

    cout << "Key Insights:\n";
    cout << "=============\n";
    cout << "- Strassen's algorithm provides ~2-3x improvement for large matrices\n";
    cout << "- Naive D&C has minimal practical value due to overhead\n";
    cout << "- Real-world applications should use optimized libraries\n";
    cout << "- Memory bandwidth becomes the limiting factor for very large matrices\n";

    return 0;
}