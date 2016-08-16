/*
 * SawtoothNetwork.h
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#ifndef LSTMNETWORK_H_
#define LSTMNETWORK_H_

#include <vector>
#include <omp.h>
#include "MemoryBlock.h"
#include "Neuron.h"
using namespace std;

class LSTMNetwork {
private:
	int inputSize;
	int timestep;
	double learningRate;
	double decayRate;
	vector<vector<MemoryBlock> > blocks;
	vector<vector<Neuron> > layers;
	vector<vector<double> > error;
	vector<double> timeSteps;
	int getPreviousNeurons();
public:
	LSTMNetwork(int is, double l, double d);
	virtual ~LSTMNetwork();
	void clear();
	void backward();
	vector<double> forward(vector<double> input);
	vector<double> forward(vector<double> input, vector<double> target);
	void addLayer(int size);
	void addLSTMLayer(int size, int cells);
};

#endif /* LSTMNETWORK_H_ */
