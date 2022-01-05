#include "NeuralNet.hpp"
#include <cstdlib>
#include <hls_math.h>

/**	Perform a copy through the AXI-4 interface. This implementation
 * 	makes use of the interface burst option via the HLS PIPELINE
 * 	directive
 *
 * 	@param [out] to
 * 		the base addres to where the data will be copied to
 * 	@param [in] from
 * 		the base address of the data origin, i.e., where the data
 * 		is coming from
 * 	@param [in] size
 * 		the number of data slots to be copied (note that it's not
 * 		the number of bytes, but the number of array positions in
 * 		accordance to the datatype)
 */
void axi4_copy(volatile datatype* to, volatile datatype* from, size_t size){
	for(int i = 0; i < size; i++)
		#pragma HLS PIPELINE II=1
		to[i] = from[i];
}


void matrix_times_vector(){}

/** This accelerator is specialized to the size of the trained neural
 * 	network, i.e., It's currently not possible to extend the number of layers.
 * 	The function applies the neural net to a set of inputs, presented by the
 * 	pointer `input_cases`
 *
 * 	At maximum, this accelerator can process the results of MAX_NUMBER_OF_INPUTS
 * 	input entries at once (This value is variable, depending on the FPGA and
 * 	Neural Net size)
 *
 * 	The neural net follows a simple MLP architecture, with a single hidden layer.
 * 	The input size is fixed to 28*28 as it's the standard image size
 *
 * */
void feed_forward(
	unsigned int i_number_of_input_cases,
	volatile datatype* i_input_cases,
	volatile datatype* i_weights_0,
	volatile datatype* i_biases_0,
	volatile datatype* i_weights_1,
	volatile datatype* i_biases_1,
	volatile datatype* o_outputs
){
	const size_t number_of_input_cases = i_number_of_input_cases;
	// Number of input test cases * the size of each input
	datatype inputs[MAX_NUMBER_OF_INPUTS*INPUT_LAYER_SIZE];
	// Layer 0
	datatype w0[INPUT_LAYER_SIZE*HIDDEN_LAYER_SIZE];
	datatype b0[HIDDEN_LAYER_SIZE];
	datatype layer_output_0[HIDDEN_LAYER_SIZE];
	// Layer 1
	datatype w1[HIDDEN_LAYER_SIZE*OUTPUT_LAYER_SIZE];
	datatype b1[OUTPUT_LAYER_SIZE];
	datatype layer_output_1[OUTPUT_LAYER_SIZE];
	// Output size: number of input test cases * the size of each output
	datatype outputs[MAX_NUMBER_OF_INPUTS*OUTPUT_LAYER_SIZE];
	// Stores the output of every layer, after passing through the
	// weights, biases and activation function. As such, it must
	// have the largest possible size for output layers, i.e. the
	// hidden layer size
	datatype layer_output_buffer[HIDDEN_LAYER_SIZE];


	{ // READ THE INPUTS TO BRAMs
		#pragma HLS INLINE
		// Copying the input data for all test cases
		axi4_copy(inputs, i_input_cases, number_of_input_cases*INPUT_LAYER_SIZE);

		// Copy layer data for layer 0
		axi4_copy(w0, i_weights_0, INPUT_LAYER_SIZE*HIDDEN_LAYER_SIZE);
		axi4_copy(b0, i_biases_0, HIDDEN_LAYER_SIZE);

		// Copy layer data for layer 0
		axi4_copy(w1, i_weights_1, HIDDEN_LAYER_SIZE*OUTPUT_LAYER_SIZE);
		axi4_copy(b1, i_biases_1, OUTPUT_LAYER_SIZE);
	}

	for(size_t input_case = 0; input_case < number_of_input_cases; input_case++){
		// For each input case presented

		// PROCESS INPUT THROUGH EACH LAYER

		// SAVE RESULTS TO THE CORRECT OUTPUT
	}

	{ // COPY LOCAL OUTPUT IN BRAM TO MEMORY
		#pragma HLS INLINE
		// Copying back the processed output for all test cases
		axi4_copy(o_outputs, outputs, number_of_input_cases*OUTPUT_LAYER_SIZE);
	}
}
