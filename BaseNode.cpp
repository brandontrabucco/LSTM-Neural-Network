/*
 * Neuron.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "BaseNode.h"

BaseNode::BaseNode() {
	// TODO Auto-generated constructor stub
}

BaseNode::~BaseNode() {
	// TODO Auto-generated destructor stub
}

double BaseNode::sigmoid(double input) {
	return 1 / (1 + exp(-input));
}

double BaseNode::sigmoidPrime(double input) {
	return sigmoid(input) * (1 - sigmoid(input));
}

double BaseNode::activationFunction(double input) {
	return tanh(input);
}

double BaseNode::activationFunctionPrime(double input) {
	return (1 - (tanh(input) * tanh(input)));
}

