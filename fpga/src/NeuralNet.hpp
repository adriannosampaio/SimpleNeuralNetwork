#pragma once

// Defining the base dimensions of the network
#define NUMBER_OF_LAYERS 3
#define INPUT_LAYER_SIZE 28*28
#define HIDDEN_LAYER_SIZE 30
#define OUTPUT_LAYER_SIZE 10
#define COORD_2D_TO_1D(I,J,N_COLS) I*N_COLS+J

// Accelerator Properties
#define MAX_NUMBER_OF_INPUTS 100
#define MAX_INPUT_MEMORY INPUT_LAYER_SIZE*MAX_NUMBER_OF_INPUTS

// Accelerator Datatype
typedef float datatype;


void feed_forward(
	unsigned int i_number_of_input_cases,
	volatile datatype* i_input_cases,
	volatile datatype* i_weights_0,
	volatile datatype* i_biases_0,
	volatile datatype* i_weights_1,
	volatile datatype* i_biases_1,
	volatile datatype* o_outputs
);
