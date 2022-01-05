import json
import sys

class Layer:
	def __init__(self, n_inputs, n_outputs):
		self.n_inputs = n_inputs								
		self.n_outputs = n_outputs					
		self.weights = None
		self.biases = None


def read_model_definition(filename):
	layers = []
	file = open(filename, "r")
	json_file = json.loads(file.read())
	for layer_data in json_file['layers']:
		l = Layer(layer_data['num_input_layers'], layer_data['num_output_layers'])
		print(layer_data['biases'])
		l.biases = layer_data['biases']
		l.weights = layer_data['weights']
		layers.append(l)
	return layers

def main():
	layers = read_model_definition(sys.argv[1])
	f = open('output.txt', 'w')
	for l in layers:
		for b in l.biases:
			f.write(f'{b} ')
		f.write('\n')
		
		for w in l.weights:
			f.write(f'{w} ')
		f.write('\n')

if __name__ == '__main__':
	main()