#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4819)

#include <iostream>
#include <string>
#include <random>
#include <sstream>
#include <iomanip>
#include <numeric>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <opencv2/opencv.hpp>

cv::Mat capture(const cv::Mat& prj_im)
{
	//
	// Write your code for projection a projector image in "prj_im" and capture a camera image to "cam_im"
	//

	// This is dummy
	cv::Mat cam_im = prj_im.clone();

	return cam_im;
}

struct fitting_functor
{
	fitting_functor(const int inputs, const int values, const Eigen::VectorXd& input_x, const Eigen::VectorXd& input_y, const int im_num_)
		: inputs_(inputs), values_(values), im_num(im_num_), N(values / im_num_), x(input_x), y(input_y) {}

	Eigen::VectorXd x;
	Eigen::VectorXd y;

	int operator()(const Eigen::VectorXd& p, Eigen::VectorXd& fvec) const
	{
		// f = min(1, c_i * ((a * x + b)^g) + d_i);
		// p[0] --- p[N-1] : c_i
		// p[N] --- p[2*N-1] : d_i
		// p[2*N+0] : a
		// p[2*N+1] : b
		// p[2*N+2] : g

		for (int pix_idx = 0; pix_idx < N; ++pix_idx)
		{
			for (int im_idx = 0; im_idx < im_num; ++im_idx)
			{
				const int elem_idx = im_num * pix_idx + im_idx;

				const auto& c_i = p[pix_idx];
				const auto& d_i = p[N + pix_idx];
				const auto& a = p[2 * N + 0];
				const auto& b = p[2 * N + 1];
				const auto& g = p[2 * N + 2];

				const auto& x_ = x[im_idx];
				const auto& y_ = y[elem_idx];

				fvec[elem_idx] = std::pow(std::min(1.0, c_i * std::pow(std::max(0.0, a * x_ + b), g) + d_i) - y_, 2);
			}
		}
		return 0;
	}

	int df(const Eigen::VectorXd& p, Eigen::MatrixXd& fjac)
	{
		fjac.setZero();

		for (int pix_idx = 0; pix_idx < N; ++pix_idx)
		{
			for (int im_idx = 0; im_idx < im_num; ++im_idx)
			{
				const int elem_idx = im_num * pix_idx + im_idx;

				const auto& c_i = p[pix_idx];
				const auto& d_i = p[N + pix_idx];
				const auto& a = p[2 * N + 0];
				const auto& b = p[2 * N + 1];
				const auto& g = p[2 * N + 2];

				const auto& x_ = x[im_idx];
				const auto& y_ = y[elem_idx];

				const double E = std::max(0.0, a * x_ + b);
				const double D = std::pow(E, g);
				const double C = c_i * D + d_i;
				const double B = std::min(1.0, C);
				const double A = B - y_;

				const double uC = C > 1.0 ? 1.0 : 0.0;

				const double df_dd = 2.0 * A * (1.0 - uC);
				const double df_dc = df_dd * D;
				const double df_dg = E > 0.0 ? df_dc * c_i * std::log(E) : 0.0;
				const double df_db = df_dd * c_i * g * std::pow(E, g - 1);
				const double df_da = df_db * x_;
				
				fjac(elem_idx, pix_idx) = df_dc;
				fjac(elem_idx, N + pix_idx) = df_dd;
				fjac(elem_idx, 2 * N + 0) = df_da;
				fjac(elem_idx, 2 * N + 1) = df_db;
				fjac(elem_idx, 2 * N + 2) = df_dg;
			}
		}

		return 0;
	}

	const int inputs_;
	const int values_;
	const int N;
	const int im_num;
	int inputs() const { return inputs_; }
	int values() const { return values_; }
};

double opt_func(const int N, const int im_num, const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& p)
{
	double error = 0.0;

	for (int pix_idx = 0; pix_idx < N; ++pix_idx)
	{
		for (int im_idx = 0; im_idx < im_num; ++im_idx)
		{
			const int elem_idx = im_num * pix_idx + im_idx;

			const auto& c_i = p[pix_idx];
			const auto& d_i = p[N + pix_idx];
			const auto& a = p[2 * N + 0];
			const auto& b = p[2 * N + 1];
			const auto& g = p[2 * N + 2];

			const auto& x_ = x[im_idx];
			const auto& y_ = y[elem_idx];

			error += std::pow(std::min(1.0, c_i * std::pow(a * x_ + b, g) + d_i) - y_, 2);
		}
	}

	return error;
}

Eigen::VectorXd opt_func_df(const int N, const int im_num, const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& p)
{
	Eigen::VectorXd df = Eigen::VectorXd::Zero(2 * N + 3);

	for (int pix_idx = 0; pix_idx < N; ++pix_idx)
	{
		for (int im_idx = 0; im_idx < im_num; ++im_idx)
		{
			const int elem_idx = im_num * pix_idx + im_idx;

			const auto& c_i = p[pix_idx];
			const auto& d_i = p[N + pix_idx];
			const auto& a = p[2 * N + 0];
			const auto& b = p[2 * N + 1];
			const auto& g = p[2 * N + 2];

			const auto& x_ = x[im_idx];
			const auto& y_ = y[elem_idx];

			const double E = std::max(0.0, a * x_ + b);
			const double D = std::pow(E, g);
			const double C = c_i * D + d_i;
			const double B = std::min(1.0, C);
			const double A = B - y_;

			const double uC = C > 1.0 ? 1.0 : 0.0;

			const double df_dd = 2.0 * A * (1.0 - uC);
			const double df_dc = df_dd * D;
			const double df_dg = E > 0.0 ? df_dc * c_i * std::log(E) : 0.0;
			const double df_db = df_dd * c_i * g * std::pow(E, g - 1);
			const double df_da = df_db * x_;

			df[pix_idx] += df_dc;
			df[N + pix_idx] += df_dd;
			df[2 * N + 0] += df_da;
			df[2 * N + 1] += df_db;
			df[2 * N + 2] += df_dg;
		}
	}

	return df;
}

int main()
{
	//
	// modify these arguments for your usage
	//
	const int im_num = 25;
	const cv::Size cam_resized_size(8, 4);
	const cv::Size prj_size(640, 480);
	const std::string data_path = "./";


	Eigen::VectorXd prj_intensities = Eigen::VectorXd::Zero(im_num);
	for (int i = 0; i < im_num; ++i)
	{
		prj_intensities[i] = static_cast<double>(i) / (im_num - 1);
	}

	std::vector<cv::Mat> cam_im_vec;


	for (int i = 0; i < im_num; ++i)
	{
		std::cout << prj_intensities[i] * 255.0 << std::endl;

		const cv::Mat prj_im = cv::Mat::ones(prj_size, CV_8U) * static_cast<int>(prj_intensities[i] * 255.0);
		//const cv::Mat prj_im = cv::Mat::ones(setting.prj.size, CV_8U) * static_cast<int>(prj_intensities[i]);
		const cv::Mat cam_im = capture(prj_im);
		cv::imwrite(data_path + std::to_string(i) + "_org.bmp", cam_im);

		cv::Mat cam_im_resized;
		cv::resize(cam_im, cam_im_resized, cam_resized_size);

		cam_im_vec.push_back(cam_im_resized);

		cv::imshow("cam_im_resized", cam_im_resized);
		cv::waitKey(1);

		cv::imwrite(data_path + std::to_string(i) + ".bmp", cam_im_resized);
	}

	const int N = cam_resized_size.area();
	//const int N = 1;

	// y[0*im_num] --- p[1*im_num-1] : for cam_idx=0
	// y[1*im_num] --- p[2*im_num-1] : for cam_idx=1
	// ---
	// y[(N-1)*im_num] --- p[N*im_num-1] : for cam_idx=N
	Eigen::VectorXd cam_intensities_vec = Eigen::VectorXd::Zero(N * im_num);

	for (int cam_y = 0; cam_y < cam_resized_size.height; ++cam_y)
	{
		std::vector<uchar*> cam_ptrs(im_num);
		for (int i = 0; i < im_num; ++i)
		{
			cam_ptrs[i] = cam_im_vec[i].ptr<uchar>(cam_y);
		}

		for (int cam_x = 0; cam_x < cam_resized_size.width; ++cam_x)
		{
			const int cam_idx = cam_y * cam_resized_size.width + cam_x;

			for (int i = 0; i < im_num; ++i)
			{
				const auto val = cam_ptrs[i][cam_x];
				cam_intensities_vec[cam_idx * im_num + i] = static_cast<double>(val) / 255.0;
			}
		}
	}

	// f = min(1, c_i * ((a * x + b)^g) + d_i);
	// p[0] --- p[N-1] : c_i
	// p[N] --- p[2*N-1] : d_i
	// p[2*N+0] : a
	// p[2*N+1] : b
	// p[2*N+2] : g

	Eigen::VectorXd p(2 * N + 3);
	
	for (int i = 0; i < N; ++i)
	{
		p[i] = 1.0;
		p[N+i] = 0.0;
	}
	p[2 * N + 0] = 1.0;
	p[2 * N + 1] = 0.0;
	p[2 * N + 2] = 1.0;

	fitting_functor functor(2 * N + 3, N * im_num, prj_intensities, cam_intensities_vec, im_num);
	Eigen::LevenbergMarquardt<fitting_functor> lm(functor);

	Eigen::LevenbergMarquardtSpace::Status info = lm.minimize(p);

	std::cout << "lm.fnorm:" << std::endl;
	std::cout << lm.fnorm << std::endl;

	const Eigen::VectorXd c_vec = p.block(0, 0, N, 1);
	const Eigen::VectorXd d_vec = p.block(N, 0, N, 1);

	double min_val = std::numeric_limits<double>::max();
	int min_idx = 0;
	for (int i = 0; i < N; ++i)
	{
		const double c = c_vec[i];
		const double d = d_vec[i];
		const double a = p[2 * N + 0];
		const double b = p[2 * N + 1];
		const double g = p[2 * N + 2];

		double x = (std::pow((1.0 - d) / c, 1.0 / g) - b) / a;
		x = std::isnan(x) ? std::numeric_limits<double>::max() : x;

		if (x < min_val)
		{
			min_idx = i;
			min_val = x;
		}
	}

	//std::cout << c_vec.mean() << std::endl;
	//std::cout << d_vec.mean() << std::endl;
	//std::cout << c_vec[min_idx] << std::endl;
	//std::cout << d_vec[min_idx] << std::endl;
	//std::cout << min_val << std::endl;
	//std::cout << p.tail(3) << std::endl;
	//std::cout << info << std::endl;

	std::cout << "c_thr: " << c_vec[min_idx] << std::endl;
	std::cout << "d_thr: " << d_vec[min_idx] << std::endl;
	std::cout << "prj_thr: " << min_val << std::endl;
	std::cout << "a: " << p[2 * N + 0] << std::endl;
	std::cout << "b: " << p[2 * N + 1] << std::endl;
	std::cout << "g: " << p[2 * N + 2] << std::endl;
	std::cout << "k: " << (1.0 - d_vec[min_idx]) / c_vec[min_idx] << std::endl;

	/*
	cv::FileStorage fs(setting.optical_calibration_path, cv::FileStorage::WRITE);

	if (!fs.isOpened())
	{
		std::cout << "Error: Failed to open calibration file." << std::endl;
		throw std::runtime_error("optical_calib_open");
	}

	cv::write(fs, "a", p[2 * N + 0]);
	cv::write(fs, "b", p[2 * N + 1]);
	cv::write(fs, "gamma", p[2 * N + 2]);
	cv::write(fs, "k", (1.0 - d_vec[min_idx]) / c_vec[min_idx]);

	fs.release();
	*/

	cv::waitKey(-1);

	return 0;
}