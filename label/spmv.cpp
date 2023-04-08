// On Linux and MacOS, you can compile and run this program like so:
//   g++ -std=c++11 -O3 -DNDEBUG -DTACO -I ../include -L../build/lib spmv.cpp -o spmv -ltaco
//   LD_LIBRARY_PATH=../build/lib ./spmv
//   gdb --args /home/rubing/taco/tools/
//   file spmv
//   run file spmv
#include <random>
#include <ctime>
#include <vector>
#include <fstream>
#include <string>
#include "taco.h"

using namespace std;
using namespace taco;

int main(int argc, char* argv[]) {
    default_random_engine gen(0);
    uniform_real_distribution<double> unif(0.0, 1.0);

    Format DV({Dense});
    Format DM({Dense, Dense});
    Format format_list[] = { DM, COO(2), CSR, CSC, DCSR, DCSC };
    string format_names[] = { "DM", "COO", "CSR", "CSC", "DCSR", "DCSC" };

    ifstream filename_list("./data/suite-sparse/small_file_list.txt");
    ofstream spmv_out_file("./data/operator/spmv_out.txt", ios_base::app);
    ofstream spmv_error_file("./data/operator/spmv_error.log", ios_base::app);

    string filename;
    while (filename_list >> filename) {
        cout << filename << endl;
        try {
            Tensor<double> temp_A = read(filename, CSR, false);
            Tensor<double> x({temp_A.getDimension(1)}, DV);
            Tensor<double> y({temp_A.getDimension(0)}, DV);

            for (int i = 0; i < x.getDimension(0); i++) {
                x.insert({i}, unif(gen));
            }
            x.pack();
            string out_line = "";

            out_line += filename + " ";
            for (int i = 0; i < 6; i++) {
                Format format = format_list[i];
                string format_name = format_names[i];
                cout << format_name << " read" << endl;
                Tensor<double> A = read(filename, format, false);
                cout << format_name << " pack" << endl;
                A.pack();

                for (int j = 0; j < 3; j++) {
                    IndexVar var_i("var_i"), var_j("var_j");
                    // y(var_i) = A(var_i, var_j) * x(var_j);
                    Access matrix = A(var_i, var_j);
                    y(var_i) = matrix * x(var_j);

                    IndexStmt stmt = y.getAssignment().concretize();
                    TensorVar workspace("workspace", Type(Float64, {Dimension(A.getDimension(1))}), Dense);
                    stmt = stmt.precompute(A(var_i, var_j) * x(var_j), var_j, var_j, workspace);

                    cout << format_name << " compile" << endl;
                    y.compile();
                    y.assemble();
                    clock_t start = clock();
                    y.compute();
                    clock_t end = clock();
                    double secends = (end - start) / (double)CLOCKS_PER_SEC;
                    cout << format_name << " secends: " << secends << endl;
                    out_line += to_string(secends) + " ";
                }
            }
            spmv_out_file << out_line << endl;
        } catch (exception &e) {
            spmv_error_file << filename << " " << e.what() << endl;
        }
    }
}