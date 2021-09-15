#pragma once

#include "MnistReader.hpp"
// Eigen
// Standard Library C++
#include <vector>
#include <fstream>
#include <string_view>
#include <iomanip>
#include <iostream>
#include <ios>
#include <map>

namespace mnist2eigen
{

	int Internal::reverse_integer(int value)
	{
		unsigned char c1, c2, c3, c4;
		c1 = value & 255;
		c2 = (value >> 8) & 255;
		c3 = (value >> 16) & 255;
		c4 = (value >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	}



	MNISTImageReader::MNISTImageReader(const std::string& filename)
	{
		m_file = std::ifstream(filename, std::ifstream::binary);
		if (m_file.is_open())
		{
			read_magic_number();
			read_dimensions();
			read_data();
		}
		m_file.close();
	}

	void MNISTImageReader::read_dimensions(){
		m_file.read((char*)&m_number_of_images, sizeof(m_number_of_images));
		m_number_of_images = Internal::reverse_integer(m_number_of_images);
		m_file.read((char*)&m_number_of_rows, sizeof(m_number_of_rows));
		m_number_of_rows = Internal::reverse_integer(m_number_of_rows);
		m_file.read((char*)&m_number_of_cols, sizeof(m_number_of_cols));
		m_number_of_cols = Internal::reverse_integer(m_number_of_cols);
			
		std::cout 
			<< "dt = " << (int) (m_magic_number & 255) 
			<< " " << (int) ((m_magic_number >> 8) & 255) << "\n\n";
	}

	void MNISTImageReader::read_magic_number()
	{
		m_file.read((char*)&m_magic_number, sizeof(m_magic_number));
		m_magic_number = Internal::reverse_integer(m_magic_number);
	}

	void MNISTImageReader::read_data(){
		int data_size = m_number_of_images * m_number_of_cols * m_number_of_rows;
		std::vector<uint8_t> data(data_size);
		m_file.read(reinterpret_cast<char*>(&(data[0])), data_size);

		int img_size = m_number_of_rows * m_number_of_cols;
		int row_size = m_number_of_cols;
		m_data.resize(m_number_of_images, m_number_of_rows * m_number_of_cols);
		for (int img = 0; img < m_number_of_images; img++)
		{
			for (int r = 0; r < m_number_of_rows; r++)
			{
				for (int c = 0; c < m_number_of_cols; c++)
				{
					// Remember that each image must be flatten before, so
					// the matrix is number_of_images x number_of_pixels
					m_data(img, r * row_size + c) =
						(double) data[img * img_size + r * row_size + c] / 255;
				}
			}
		}
	}

	const Eigen::MatrixXd& MNISTImageReader::get_data()
	{
		return this->m_data;
	}

	MNISTLabelReader::MNISTLabelReader(const std::string& filename)
	{
		m_file = std::ifstream(filename, std::ifstream::binary);
		if (m_file.is_open()){
			read_magic_number();
			read_dimensions();
			read_data();
		}
		else {
			std::string msg = "Error: could not find file: " + filename + ".\n";
			std::cerr << msg;
			throw std::runtime_error(msg);
		}
	}

	void MNISTLabelReader::read_dimensions(){
		m_file.read((char*)&m_number_of_images, sizeof(m_number_of_images));
		m_number_of_images = Internal::reverse_integer(m_number_of_images);
		std::cout 
			<< "dt = " << (int) (m_magic_number & 255) 
			<< " " << (int) ((m_magic_number >> 8) & 255) << "\n\n";
	}

	void MNISTLabelReader::read_magic_number()
	{
		m_file.read((char*)&m_magic_number, sizeof(m_magic_number));
		m_magic_number = Internal::reverse_integer(m_magic_number);
	}

	void MNISTLabelReader::read_data(){
		int data_size = m_number_of_images;
		std::vector<uint8_t> data(data_size);
		m_file.read(reinterpret_cast<char*>(&(data[0])), data_size);

		m_data.resize(m_number_of_images, 1);
		for (int img = 0; img < m_number_of_images; img++)
		{
			// Remember that each image must be flatten before, so
			// the matrix is number_of_images x number_of_pixels
			m_data(img, 0) = data[img];
		}
	}

	const Eigen::MatrixXi& MNISTLabelReader::get_data()
	{
		return m_data;
	}


	void write_ppm(const std::string& filename, const Eigen::MatrixXd& data, int row) {
		std::ofstream file(filename, std::fstream::binary);
		file << 'P' << '6' << '\n'
			<< '2' << '8' << ' ' << '2' << '8' << '\n'
			<< '2' << '5' << '5' << '\n';
		for (int i = 0; i < 28; i++)
		{
			for (int j = 0; j < 28; j++)
			{
				int val = data(row, i*28+ j) * 255;
				char data = (char)val;
				for(int p = 0; p < 3; p++)
					file.put(data);
			}
		}
	}

	MNISTData read_mnist_dataset(const std::string& folder)
	{
		const static std::string train_images_file = "train-images-idx3-ubyte";
		const static std::string train_labels_file = "train-labels-idx1-ubyte";
		const static std::string test_images_file  = "t10k-images-idx3-ubyte";
		const static std::string test_labels_file  = "t10k-labels-idx1-ubyte";

		MNISTImageReader test_images (folder + "/"  + test_images_file);
		MNISTLabelReader test_labels (folder + "/"  + test_labels_file);
		MNISTImageReader train_images(folder + "/" + train_images_file);
		MNISTLabelReader train_labels(folder + "/" + train_labels_file);

		MNISTData data;
		data.test_images = test_images.get_data();
		data.test_labels = test_labels.get_data();
		data.train_images = train_images.get_data();
		data.train_labels = train_labels.get_data();

		return data;
	}

}