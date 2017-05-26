#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	unsigned long number_of_estimations = estimations.size();

	if(number_of_estimations != ground_truth.size() || number_of_estimations == 0) {
		std::cout << "Invalid estimation or ground_truth data" << std::endl;
	    return rmse;
	}

	//accumulate squared residuals
	for (unsigned int i = 0; i < number_of_estimations; ++i) {
		VectorXd residual = estimations[i] - ground_truth[i];

	    //coefficient-wise multiplication
	    residual = residual.array() * residual.array();
	    rmse += residual;
	}

	//calculate the mean
	rmse = rmse / number_of_estimations;

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}
