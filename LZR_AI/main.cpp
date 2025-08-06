#include <iostream>
#include "CSVModule.h"

int main()
{
	CCSVModule csvModule;

	// CSV 파일 경로
	std::string filePath = "data/sample.csv";

	// 헤더 정의
	std::vector<std::string> header;
	for (int idx = 0; idx < 274; ++idx) {
		header.push_back("Plane0_" + std::to_string(idx));
	}
	for (int idx = 0; idx < 274; ++idx) {
		header.push_back("Plane1_" + std::to_string(idx));
	}
	for (int idx = 0; idx < 274; ++idx) {
		header.push_back("Plane2_" + std::to_string(idx));
	}
	for (int idx = 0; idx < 274; ++idx) {
		header.push_back("Plane3_" + std::to_string(idx));
	}
	header.push_back("HUMAN");

	// CSV 파일 생성 (헤더 포함)
	csvModule.createCSVwithHeader(filePath, header);

	// 데이터 행 추가
	std::vector<std::string> row1 = { "Alice", "30", "New York" };
	csvModule.appendToCSV(filePath, row1);

	std::vector<std::string> row2 = { "Bob", "25", "Los Angeles" };
	csvModule.appendToCSV(filePath, row2);

	// CSV 파일의 헤더 읽기
	/*
	std::vector<std::string> readHeader = csvModule.readCSVHeader(filePath);
	std::cout << "CSV Header: ";
	for (const auto& col : readHeader) {
		std::cout << col << " ";
	}
	std::cout << std::endl;

	// CSV 파일의 전체 내용 읽기
	auto data = csvModule.readCSVFile(filePath);
	std::cout << "CSV Data:" << std::endl;
	for (const auto& row : data) {
		for (const auto& col : row) {
			std::cout << col << " ";
		}
		std::cout << std::endl;
	}
	*/
	return 0;

}
