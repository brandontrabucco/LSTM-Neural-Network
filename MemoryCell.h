/*
 * MemoryCell.h
 *
 *  Created on: Jul 18, 2016
 *      Author: trabucco
 */

#ifndef MEMORYCELL_H_
#define MEMORYCELL_H_

#include "BaseNode.h"
#include <vector>
#include <math.h>
#include <time.h>
#include <iostream>
#include <random>
using namespace std;

class MemoryCell : BaseNode {
private:
	static long long n;
public:
	vector<double> cellDataWeight, cellDataPartial,
		inputDataPartial, forgetDataPartial;
	double cellFeedbackWeight, bias;
	double activationIn, activationInPrime,
		activationOut, activationOutPrime,
		state, previousState,
		feedback, previousFeedback,
		cellFeedbackPartial;
	double inputFeedbackPartial, inputStatePartial,
		forgetFeedbackPartial, forgetStatePartial;
	double activateIn(double data);
	double activateOut(double data);
	MemoryCell(int connections);
	virtual ~MemoryCell();
};

#endif /* MEMORYCELL_H_ */
