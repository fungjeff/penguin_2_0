#ifndef OUTPUT_H
#define OUTPUT_H

//=======================================================================
// File control
//=======================================================================

void open_output_file(ofstream&, string);

void append_output_file(ofstream&, string);

void open_binary_file(ofstream&, string);

void open_binary_file(ifstream&, string);

//=======================================================================
// Outputs
//=======================================================================

string create_label();

__global__ void clear_output(sdp*, sdp*, sdp*, sdp*, sdp*);

__global__ void cal_output(hydr_ring*, sdp, sdp*, sdp*, sdp*, sdp*, sdp*);

void save_cells(ofstream&, GPU_plan*);

void write_para_file(ofstream&, string&, string&);

void write_initial_file(ofstream&, sdp*, sdp*);

sdp GPU_output_reduction(GPU_plan*, body, sdp, int mode = 0);

#endif
