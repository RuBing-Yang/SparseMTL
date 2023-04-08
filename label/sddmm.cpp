// On Linux and MacOS, you can compile and run this program like so:
//   g++ -std=c++11 -O3 -DNDEBUG -DTACO -I ../include -L../build/lib sddmm.cpp -o sddmm -ltaco
//   LD_LIBRARY_PATH=../build/lib ./sddmm
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

    Format RM({Dense, Dense});
    Format CM({Dense, Dense}, {1, 0});
    Format DV({Dense});
    Format DM({Dense, Dense});
    Format format_list[] = { DM, COO(2), CSR, CSC, DCSR, DCSC };
    string format_names[] = { "DM", "COO", "CSR", "CSC", "DCSR", "DCSC" };

    ifstream filename_list("./data/suite-sparse/small_file_list.txt");
    ofstream sddmm_out_file("./data/operator/sddmm_out.txt", ios_base::app);
    ofstream sddmm_error_file("./data/operator/sddmm_error.log", ios_base::app);

    string filename;
    while (filename_list >> filename) {
        cout << filename << endl;
        string out_line = "";
        try {
            Tensor<double> temp_B = read(filename, CSR, false);
            Tensor<double> A(temp_B.getDimensions(), RM);
            Tensor<double> C({temp_B.getDimension(0), 100}, RM);
            Tensor<double> D({C.getDimension(1), temp_B.getDimension(1)}, CM);
            for (int i = 0; i < C.getDimension(0); ++i) {
                for (int j = 0; j < C.getDimension(1); ++j) {
                    C.insert({i,j}, unif(gen));
                }
            }
            C.pack();
            for (int i = 0; i < D.getDimension(0); ++i) {
                for (int j = 0; j < D.getDimension(1); ++j) {
                    D.insert({i,j}, unif(gen));
                }
            }
            D.pack();
            
            out_line += filename + " ";
            for (int i = 0; i < 6; i++) {
                Format format = format_list[i];
                string format_name = format_names[i];

                cout << format_name << " read" << endl;
                Tensor<double> B = read(filename, format, false);
                cout << format_name << " pack" << endl;
                B.pack();

                for (int j = 0; j < 3; j++) {
                    IndexVar var_i("var_i"), var_j("var_j"), var_k("var_k");
                    A(var_i, var_j) = B(var_i, var_j) * C(var_i, var_k) * D(var_k, var_j);
                    cout << format_name << " compile" << endl;
                    A.compile();
                    A.assemble();

                    clock_t start = clock();
                    A.compute();
                    clock_t end = clock();
                    double secends = (end - start) / (double)CLOCKS_PER_SEC;
                    cout << format_name << " secends: " << secends << endl;
                    out_line += to_string(secends) + " ";
                }
            }
            sddmm_out_file << out_line << endl;
        } catch (exception &e) {
            sddmm_error_file << filename << " " << e.what() << endl;
        }
    }
}