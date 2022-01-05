#pragma once
/*
template<typename T, int NROWS, int NCOLS>
struct Matrix{
	const int nrows=NROWS, ncols=NCOLS;
	datatype data[NROWS*NCOLS];

	void load_from_memory(datatype* base_addr)
	{
		for(int i = 0; i < nrows*ncols; i++)
			#pragma HLS PIPELINE II=1
			data[i] = base_addr[i];
	}
};
*/
