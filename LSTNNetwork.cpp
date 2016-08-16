/*
 * LSTMNetwork.cpp
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#include "LSTMNetwork.h"

LSTMNetwork::LSTMNetwork(int is, double l, double d) {
	// TODO Auto-generated constructor stub
	inputSize = is;
	learningRate = l;
	decayRate = d;
	timestep = -1;
}

LSTMNetwork::~LSTMNetwork() {
	// TODO Auto-generated destructor stub
}

int LSTMNetwork::getPreviousNeurons() {
	if (blocks.size() == 0 && layers.size() == 0) return inputSize;
	else return (layers.size() == 0) ? (blocks[blocks.size() - 1].size() * blocks[blocks.size() - 1][0].nCells) : layers[layers.size() - 1].size();
}

void LSTMNetwork::addLayer(int size) {
	vector<Neuron> buffer;
	for (int i = 0; i < size; i++) {
		buffer.push_back(Neuron(getPreviousNeurons()));
	} layers.push_back(buffer);
}

void LSTMNetwork::addLSTMLayer(int size, int cells) {
	vector<MemoryBlock> buffer;
	for (int i = 0; i < size; i++) {
		buffer.push_back(MemoryBlock(cells, getPreviousNeurons()));
	} blocks.push_back(buffer);
}

vector<double> LSTMNetwork::forward(vector<double> input) {
	vector<double> output(blocks[0].size() * blocks[0][0].nCells),
			connections(input.size());
	copy(input.begin(), input.end(), connections.begin());
	if (input.size() == inputSize) {
		// calculate activations from bottom up
		timestep++;
		for (int b = 0; b < blocks.size(); b++) {
			output.resize(blocks[b].size() * blocks[b][0].nCells);
			//#pragma omp parallel
			for (int i = 0; i < (blocks[b].size()); i++) {
				vector<double> activations = blocks[b][i].forward(connections, timestep);
				for (int j = 0; j < blocks[b][i].nCells; j++) {
					output[i * blocks[b][i].nCells + j] = activations[j];
				}
			}

			connections.resize(blocks[b].size() * blocks[b][0].nCells);
			copy(output.begin(), output.end(), connections.begin());
		}

		output.resize(layers[layers.size() - 1].size());
		for (int i = 0; i < layers.size(); i++) {
			vector<double> activations(layers[i].size());
			//#pragma omp parallel
			for (int j = 0; j < layers[i].size(); j++) {
				// compute the activation
				activations[j] = (layers[i][j].forward(connections));
				// if at top of network, push to output
				if (i == (layers.size() - 1)) output[j] = (activations[j]);
			}

			connections.resize(layers[i].size());
			copy(activations.begin(), activations.end(), connections.begin());
		} timestep = -1; return output;
	} else {
		cout << "Target size mismatch " << input.size() << ":" << inputSize << endl;
		timestep = -1;
		return output;
	}
}

vector<double> LSTMNetwork::forward(vector<double> input, vector<double> target) {
	vector<double> output(blocks[0].size() * blocks[0][0].nCells),
			connections(input.size());
	copy(input.begin(), input.end(), connections.begin());
	if (input.size() == inputSize) {
		// calculate activations from bottom up
		timestep++;
		for (int b = 0; b < blocks.size(); b++) {
			output.resize(blocks[b].size() * blocks[b][0].nCells);
			//#pragma omp parallel
			for (int i = 0; i < (blocks[b].size()); i++) {
				//cout << b << "-" << i << " input size: " << connections.size() << endl;
				vector<double> activations = blocks[b][i].forward(connections, timestep);
				for (int j = 0; j < blocks[b][i].nCells; j++) {
					output[i * blocks[b][i].nCells + j] = activations[j];
				}
			}

			connections.resize(blocks[b].size() * blocks[b][0].nCells);
			copy(output.begin(), output.end(), connections.begin());
		}

		output.resize(layers[layers.size() - 1].size());
		error.push_back(vector<double>(layers[layers.size() - 1].size()));
		for (int i = 0; i < layers.size(); i++) {
			vector<double> activations(layers[i].size());
			//#pragma omp parallel
			for (int j = 0; j < layers[i].size(); j++) {
				// compute the activation
				activations[j] = (layers[i][j].forward(connections));
				// if at top of network, push to output
				if (i == (layers.size() - 1)) error[timestep][j] = (activations[j] - target[j]);
			}

			connections.resize(layers[i].size());
			copy(activations.begin(), activations.end(), connections.begin());
		}
		return error[timestep];
	} else {
		cout << "Target size mismatch " << input.size() << ":" << inputSize << endl;
		return vector<double>();
	}
}

void LSTMNetwork::backward() {	// memory accumulation is here
	for (; timestep >= 0; timestep--) {
		vector<double> weightedError(layers[layers.size() - 1].size());
		copy(error[timestep].begin(), error[timestep].end(), weightedError.begin());
		for (int i = (layers.size() - 1); i >= 0; i--) {
			vector<double> errorSum(layers[i][0].nConnections, 0);
			//#pragma omp parallel
			for (int j = 0; j < layers[i].size(); j++) {
				// compute the activation
				vector<double> contribution = layers[i][j].backward(weightedError[j], learningRate, timestep, error.size());
				//#pragma omp critical
				for (int k = 0; k < layers[i][0].nConnections; k++) {
					errorSum[k] += contribution[k];
				}
			}

			weightedError.resize(layers[i][0].nConnections);
			copy(errorSum.begin(), errorSum.end(), weightedError.begin());
		} for (int b = 0; b < blocks.size(); b++) {
			vector<double> errorSum((blocks[b][0].nConnections), 0);
			//#pragma omp parallel
			for (int i = 0; i < (blocks[b].size()); i++) {
				// compute the activation
				vector<double> errorChunk(blocks[b][i].nCells);
				copy((weightedError.begin() + (blocks[b][i].nCells) * (i)), (weightedError.begin() + (blocks[b][i].nCells * (i + 1))), errorChunk.begin());
				vector<double> contribution = blocks[b][i].backward(errorChunk, learningRate, timestep, error.size());
				//#pragma omp critical
				for (int k = 0; k < blocks[b][i].nConnections; k++) {
					errorSum[k] += contribution[k];
				}
			}

			weightedError.resize(blocks[b][0].nConnections);
			//cout << "size: " << weightedError.size() << " error: " << errorSum.size() << endl;
			copy(errorSum.begin(), errorSum.end(), weightedError.begin());
		} learningRate *= decayRate;
	}
}

void LSTMNetwork::clear() {
	for (int i = (layers.size() - 1); i >= 0; i--) {
		for (int j = 0; j < layers[i].size(); j++) {
			layers[i][j].clear();
		}
	} for (int b = (blocks.size() - 1); b >= 0; b--) for (int i = 0; i < blocks[b].size(); i++) {
		blocks[b][i].clear();
	} error.clear();
	timestep = -1;
}
