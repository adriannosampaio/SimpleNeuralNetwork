#pragma once

#include <fstream>
#include <string_view>
#include <Eigen/Dense>
#include <iomanip>
#include <ios>
#include <map>

namespace mnist2eigen
{

	namespace Internal 
	{
		/** Reverse the bits in a 32-bit integer. Used
		*	to convert big-endian integers to small-endian
		*	and vice-versa.
		*
		*	@param [in] value
		*		The initial integer value to be converted
		*	@return
		*		The reverse binary value
		*/
		int reverse_integer(int value);
	}

	class MNISTImageReader {
		std::ifstream m_file;
		int m_magic_number;
		int m_number_of_rows;
		int m_number_of_cols;
		int m_number_of_images;
		Eigen::MatrixXd m_data;
	public:
		MNISTImageReader(const std::string& filename);

		void read_dimensions();

		void read_magic_number();

		void read_data();

		const Eigen::MatrixXd& get_data();
	};
	
	class MNISTLabelReader {
		std::ifstream m_file;
		int m_magic_number;
		int m_number_of_images;
		Eigen::MatrixXi m_data;
	public:
		MNISTLabelReader(const std::string& filename);
		void read_dimensions();
		void read_magic_number();
		void read_data();

		const Eigen::MatrixXi& get_data();
	};

	struct MNISTData{
		Eigen::MatrixXd train_images, test_images;
		Eigen::MatrixXi train_labels, test_labels;
	};

	MNISTData read_mnist_dataset(const std::string& folder);

	void write_ppm(const std::string& filename, const Eigen::MatrixXd& data, int row = 0);
}