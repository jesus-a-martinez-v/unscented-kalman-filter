#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

#include <chrono>
#include <thread>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::pair;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_ << 1, 1, 1, 1, 1;

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  P_ << 0.1,   0,    0,   0,   0,
        0,   0.1,    0,   0,   0,
        0,   0,    0.1,   0,   0,
        0,   0,    0,   0.1,   0,
        0,   0,    0,   0,   0.1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_ = false;

  n_x_ = 5;

  n_aug_ = 7;

  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Set weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  for (unsigned short i = 1; i < 2 * n_aug_ + 1; i++) {
	  double current_weight = 0.5 / (n_aug_ + lambda_);
	  weights_(i) = current_weight;
  }
}

UKF::~UKF() {}

double normalize_angle(const double angle) {
	double normalized_angle = angle;

	while (normalized_angle > M_PI) {
		normalized_angle -= 2 * M_PI;
	}

	while (normalized_angle < -M_PI) {
		normalized_angle += 2 * M_PI;
	}

	return normalized_angle;
}

MatrixXd generate_sigma_points(const VectorXd &x, const MatrixXd &P, int n_x, double lambda) {
	// Create sigma points matrix.
	MatrixXd X_sigma = MatrixXd(n_x, 2 * n_x + 1);

	// Calculate the square root of P
	MatrixXd P_square_root = P.llt().matrixL();

	// Set first column of sigma point matrix.
	X_sigma.col(0) = x;

	//Set remaining sigma points
	for (unsigned short i = 0; i < n_x; i++) {
		VectorXd common_term = sqrt(lambda + n_x) * P_square_root.col(i);

		X_sigma.col(i + 1) = x + common_term;
		X_sigma.col(i + 1 + n_x) = x - common_term;
	}

	//cout << "X_sigma: " << X_sigma << endl;
	return X_sigma;
}

MatrixXd generate_augmented_sigma_points(VectorXd &x, MatrixXd &P, int n_x, int n_aug, double lambda, double std_a, double std_yawdd) {
	// Augmentated state and covariance.
	VectorXd x_aug = VectorXd(n_aug);
	MatrixXd P_aug = MatrixXd(n_aug, n_aug);

	// Augment state.
	x_aug.head(n_x) = x;
	x_aug(n_aug - 2) = 0;
	x_aug(n_aug - 1) = 0;

	// Augment process covariance matrix.
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x, n_x) = P;
	P_aug(n_aug - 2, n_aug - 2) = std_a * std_a;
	P_aug(n_aug - 1, n_aug - 1) = std_yawdd * std_yawdd;

	return generate_sigma_points(x_aug, P_aug, n_aug, lambda);
}

MatrixXd sigma_point_prediction(const MatrixXd &X_sigma_augmented, int n_x, int n_aug, double delta_t) {
	MatrixXd X_sigma_prediction = MatrixXd(n_x, 2 * n_aug + 1);

	for (unsigned short i = 0; i < 2 * n_aug + 1; i++) {
		// Extract values for better readability
		double p_x = X_sigma_augmented(0, i);
		double p_y = X_sigma_augmented(1, i);
		double v = X_sigma_augmented(2, i);
		double yaw = X_sigma_augmented(3, i);
		double yawd = X_sigma_augmented(4, i);
		double nu_a = X_sigma_augmented(5, i);
		double nu_yawdd = X_sigma_augmented(6, i);

		// Predicted state values
		double px_p;
		double py_p;

		// Avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
		} else {
			px_p = p_x + v * delta_t * cos(yaw);
			py_p = p_y + v * delta_t * sin(yaw);
		}

		double v_p = v;
		//cout << "v_p before noise: " << v_p << endl;
		double yaw_p = yaw + yawd * delta_t;
		double yawd_p = yawd;

		// Add noise
		double constant_term = nu_a * delta_t * delta_t;
		px_p += 0.5 * constant_term * cos(yaw);
		py_p += 0.5 * constant_term * sin(yaw);
		v_p += constant_term / delta_t;
		//cout << "delta_t " << delta_t << endl;
		//cout << "v_p after noise: " << v_p << endl;

		yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
		yawd_p += nu_yawdd * delta_t;

		// Write predicted sigma point into right column
		X_sigma_prediction(0, i) = px_p;
		X_sigma_prediction(1, i) = py_p;
		X_sigma_prediction(2, i) = v_p;
		X_sigma_prediction(3, i) = yaw_p;
		X_sigma_prediction(4, i) = yawd_p;
	}

	//cout << "X_sigma_prediction: " << X_sigma_prediction << endl;
	return X_sigma_prediction;
}

pair<VectorXd, MatrixXd> predict_mean_and_covariance(const MatrixXd &X_sigma_prediction, int n_x, int n_aug, double lambda, const VectorXd &weights) {
	// Create vector for predicted state
	VectorXd x_predicted = VectorXd(n_x);

	// Create covariance matrix for prediction
	MatrixXd P_predicted = MatrixXd(n_x, n_x);

	// Predicted state mean
	x_predicted.fill(0.0);
	for (unsigned short i = 0; i < 2 * n_aug + 1; i++) {
		x_predicted = x_predicted + weights(i) * X_sigma_prediction.col(i);
	}

	// Predicted state covariance matrix
	P_predicted.fill(0.0);
	for (unsigned short i = 0; i < 2 * n_aug + 1; i++) {
		// State difference
		VectorXd x_diff = X_sigma_prediction.col(i) - x_predicted;

		// Angle normalization
		x_diff(3) = normalize_angle(x_diff(3));

		P_predicted = P_predicted + weights(i) * x_diff * x_diff.transpose();
	}

	return std::make_pair(x_predicted, P_predicted);
}

VectorXd unpack_laser_measurement(MeasurementPackage meas_package) {
	VectorXd x = VectorXd(5);

	double p_x = meas_package.raw_measurements_[0];
	double p_y = meas_package.raw_measurements_[1];

	x << p_x, p_y, 0, 0, 0;

	return x;
}

VectorXd unpack_radar_measurement(MeasurementPackage meas_package) {
	VectorXd x = VectorXd(5);

	// First, we need to convert from polar to cartesian.
	double rho = meas_package.raw_measurements_[0];
	double phi = meas_package.raw_measurements_[1];
	double rho_dot = meas_package.raw_measurements_[2];

	double v_x = rho_dot * cos(phi);
	double v_y = rho_dot * sin(phi);
	double sqrt_of_vx_plus_vy = sqrt(v_x * v_x + v_y * v_y);

	double p_x = rho * cos(phi);
	double p_y = rho * sin(phi);

	x << p_x, p_y, sqrt_of_vx_plus_vy, 0, 0;

	return x;
}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

	if (!is_initialized_) {

		if(meas_package.sensor_type_ == meas_package.LASER) {
			x_ = unpack_laser_measurement(meas_package);
		} else if (meas_package.sensor_type_ == meas_package.RADAR) {
			x_ = unpack_radar_measurement(meas_package);
		} else {
			cout << "Received measurement from unexpected source (i.e., not RADAR nor LIDAR)" << endl;
		}

		time_us_ = meas_package.timestamp_;
		is_initialized_ = true;
	} else {
		// Normal flow
		double delta_t;
		if(meas_package.sensor_type_ == meas_package.LASER) {
			//cout << "LASER" << endl;
			x_ = unpack_laser_measurement(meas_package);
			//cout << x_ << endl;

			delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
			time_us_ = meas_package.timestamp_;

			Prediction(delta_t);
			//cout << "PREDICTION" << endl;
			//cout << "X: " << x_ << endl;
			//cout << "P: " << P_ << endl;
			//cout << "-------------------------------\n" << endl;
			UpdateLidar(meas_package);
			//cout << "UPDATE" << endl;
			//cout << "X: " << x_ << endl;
			//cout << "P: " << P_ << endl;
			//cout << "-------------------------------\n" << endl;
		} else if (meas_package.sensor_type_ == meas_package.RADAR) {
			//cout << "RADAR" << endl;
			x_ = unpack_radar_measurement(meas_package);
			//cout << x_ << endl;

			delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
			time_us_ = meas_package.timestamp_;

			Prediction(delta_t);
			//cout << "PREDICTION" << endl;
			//cout << "X: " << x_ << endl;
			//cout << "P: " << P_ << endl;
			//cout << "-------------------------------\n" << endl;
			UpdateRadar(meas_package);
			//cout << "UPDATE" << endl;
			//cout << "X: " << x_ << endl;
			//cout << "P: " << P_ << endl;
			//cout << "-------------------------------\n" << endl;
		} else {
			cout << "Received measurement from unexpected source (i.e., not RADAR nor LIDAR)" << endl;
		}

		// std::this_thread::sleep_for(std::chrono::milliseconds(5000));
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	MatrixXd X_sigma_augmented = generate_augmented_sigma_points(x_, P_, n_x_, n_aug_, lambda_, std_a_, std_yawdd_);
	Xsig_pred_ = sigma_point_prediction(X_sigma_augmented, n_x_, n_aug_, delta_t);
	pair<VectorXd, MatrixXd> predicted_mean_and_covariance = predict_mean_and_covariance(Xsig_pred_, n_x_, n_aug_, lambda_, weights_);

	x_ = predicted_mean_and_covariance.first;
	P_ = predicted_mean_and_covariance.second;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	const short n_z = 2;

	// Create matrix for sigma points in measurement space
	MatrixXd Z_sigma = MatrixXd(n_z, 2 * n_aug_ + 1);

	// Transform sigma points into measurement space

	for (unsigned short i = 0; i < 2 * n_aug_ + 1; i++) {
		// Extract values for better readability
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);

		// Measurement model
		Z_sigma(0, i) = p_x;
		Z_sigma(1, i) = p_y;
	}


	// Mean predicted measurement
	VectorXd z_predicted = VectorXd(n_z);

	z_predicted.fill(0.0);
	for (unsigned short i = 0; i < 2 * n_aug_ + 1; i++) {
		z_predicted = z_predicted + weights_(i) * Z_sigma.col(i);
	}

	// Measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (unsigned short i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd z_diff = Z_sigma.col(i) - z_predicted;
		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	// Measurement covariance matrix (laser)
	MatrixXd R_laser = MatrixXd(2, 2);

	R_laser << std_laspx_ * std_laspx_, 0,
			   0, std_laspy_ * std_laspy_;

	S = S + R_laser;
	MatrixXd S_inverse = S.inverse();

	// Cross correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);

	for (unsigned short i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd z_diff = Z_sigma.col(i) - z_predicted;
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		x_diff(3) = normalize_angle(x_diff(3));

		Tc =  Tc + weights_(i) * x_diff * z_diff.transpose();
	}


	// Kalman gain
	MatrixXd K = Tc * S_inverse;

	VectorXd z = VectorXd(n_z);
	z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];

	VectorXd z_diff = z - z_predicted;

	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	NIS_laser_ = z_diff.transpose() * S_inverse * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	const short n_z = 3;

	// Create matrix for sigma points in measurement space
	MatrixXd Z_sigma = MatrixXd(n_z, 2 * n_aug_ + 1);

	// Transform sigma points into measurement space
	for (unsigned short i = 0; i < 2 * n_aug_ + 1; i++) {
		// Extract values for better readability
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

		double v1 = cos(yaw) * v;
		double v2 = sin(yaw) * v;

		// Measurement model
		double rho = sqrt(p_x * p_x + p_y * p_y);
		double phi = atan2(p_y, p_x);
		double rho_dot = (p_x * v1 + p_y * v2) / rho;

		Z_sigma(0, i) = rho;
		Z_sigma(1, i) = phi;
		Z_sigma(2, i) = rho_dot;
	}

	// Mean predicted measurement
	VectorXd z_predicted = VectorXd(n_z);
	z_predicted.fill(0.0);
	for (unsigned short i = 0; i < 2 * n_aug_ + 1; i++) {
		z_predicted = z_predicted + weights_(i) * Z_sigma.col(i);
	}

	// Measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (unsigned short i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd z_diff = Z_sigma.col(i) - z_predicted;

		z_diff(1) = normalize_angle(z_diff(1));

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	// Add measurement noise covariance matrix
	MatrixXd R_radar = MatrixXd(n_z, n_z);
	R_radar << std_radr_ * std_radr_, 0, 0,
			   0, std_radphi_ * std_radphi_, 0,
			   0, 0, std_radrd_ * std_radrd_;

	S = S + R_radar;
	MatrixXd S_inverse = S.inverse();

	// Create matrix for cross correlation
	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);
	for (unsigned short i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd z_diff = Z_sigma.col(i) - z_predicted;
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		// Angle normalization
		z_diff(1) = normalize_angle(z_diff(1));
		x_diff(3) = normalize_angle(x_diff(3));

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	// Kalman gain
	MatrixXd K = Tc * S_inverse;

	// Current measurement
	VectorXd z = VectorXd(n_z);
	z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];

	VectorXd z_diff = z - z_predicted;

	z_diff(1) = normalize_angle(z_diff(1));

	// Update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	NIS_radar_ = z_diff.transpose() * S_inverse * z_diff;
}
