/*
 * Neuron.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef BASENODE_H_
#define BASENODE_H_

#include <math.h>
#include <vector>
#include <iostream>
using namespace std;

class BaseNode {
private:
public:
	BaseNode();
	virtual ~BaseNode();
	double sigmoid(double input);
	double sigmoidPrime(double input);
	double activationFunction(double input);
	double activationFunctionPrime(double input);
};

#endif /* BASENODE_H_ */
