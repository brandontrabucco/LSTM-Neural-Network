/*
 * MemoryBlock.h
 *
 *  Created on: Jul 18, 2016
 *      Author: trabucco
 */

#ifndef MEMORYBLOCK_H_
#define MEMORYBLOCK_H_

#include "BaseNode.h"
#include "MemoryCell.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <random>
using namespace std;

class MemoryBlock : public BaseNode {
private:
	static long long n;
public:
	int nConnections;
	int nCells;
	vector<MemoryCell> cells;
	vector<vector<double> > impulse;	// time based
	vector<double> inputDataWeight,
		forgetDataWeight, outputDataWeight,
		bias,
		inputFeedbackWeight, inputStateWeight,
		forgetFeedbackWeight, forgetStateWeight,
		outputFeedbackWeight, outputStateWeight;
	vector<double> input, inputPrime,	// time based
		forget, forgetPrime,
		output, outputPrime;
	double inputGate(double data);
	double forgetGate(double data);
	double outputGate(double data);
	MemoryBlock(int cl, int cn);
	virtual ~MemoryBlock();
	void clear();
	vector<double> forward(vector<double> input, int t);
	vector<double> backward(vector<double> errorPrime, double learningRate, int t, int length);
};

#endif /* MEMORYBLOCK_H_ */
