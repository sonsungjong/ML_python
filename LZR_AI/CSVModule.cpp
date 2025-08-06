#include "CSVModule.h"

#include <filesystem>
#include <ctime>
#include <fstream>
#include <sstream>

// csv 을 생성하고 내용을 저장하는 모듈

CCSVModule::CCSVModule()
{
}

CCSVModule::~CCSVModule()
{
}

// 해당 경로에 폴더가 있는지 검사하고 없으면 폴더를 생성한다 (내부적으로 자동사용, 따로 필요할 경우에만 사용할 것)
void CCSVModule::createFolderIfNotExists(const std::string& folderPath)
{
	if ((!std::filesystem::exists(folderPath)) && (folderPath != "")) {
		std::filesystem::create_directories(folderPath);			// 중첩 폴더 생성 지원
	}
}

// csv파일이 있는지 검사하고 없으면 해당 헤더명을 포함해서 csv파일을 생성한다 (빈 csv를 만드려면 비어있는 벡터를 전달)
void CCSVModule::createCSVwithHeader(const std::string& filePath, const std::vector<std::string>& header)
{
	std::filesystem::path pathObj(filePath);
	std::string strFolderPath = pathObj.parent_path().string();				// 파일 경로에서 폴더 경로만 추출

	createFolderIfNotExists(strFolderPath);				// 해당 폴더가 있는지 먼저 검사하고 없으면 만든다

	if (!std::filesystem::exists(filePath)) {
		std::ofstream file(filePath);
		if (file.is_open()) {
			if (!header.empty()) { // 헤더가 비어 있지 않을 때만 작성
				for (size_t i = 0U; i < header.size(); ++i) {
					file << header[i];
					if (i < header.size() - 1) {
						file << ","; // 열 구분
					}
				}
				file << "\n"; // 헤더 이후 줄바꿈 추가
			}
			// header가 비어 있으면 아무 내용도 쓰지 않고 빈 파일만 생성
			file.close();
		}
	}
}

// 한개의 행을 csv에 내용을 추가한다
void CCSVModule::appendToCSV(const std::string& filePath, const std::vector<std::string>& row)
{
	std::lock_guard<std::mutex> lock(m_mtxCSV);

	std::string strFolderPath = "";
	createFolderIfNotExists(strFolderPath);									// 폴더가 있는지 먼저 검사한다

	std::ofstream file(filePath, std::ios::app);						// append 모드 (해당 폴더경로에 파일없으면 자동생성)
	if (file.is_open()) {
		if (!row.empty()) {
			for (size_t i = 0U; i < row.size(); ++i) {
				file << row[i];
				if (i < row.size() - 1) {
					file << ",";						// 열 구분
				}
			}
			file << "\n";						// 행 구분
		}
		file.close();
	}
}

// 해당 csv의 첫 행 정보를 가져온다
std::vector<std::string> CCSVModule::readCSVHeader(const std::string& filePath)
{
	std::vector<std::string> vecHeaderData;
	std::ifstream file(filePath);

	if (file.is_open()) {
		std::string line = "";
		if (std::getline(file, line)) {
			std::stringstream ss(line);
			std::string column;

			while (std::getline(ss, column, ','))					// ',' 를 기준으로 열을 구분한다
			{
				vecHeaderData.push_back(column);			// 각 컬럼을 벡터에 추가
			}
		}

		file.close();
	}

	return vecHeaderData;
}

std::vector<std::vector<std::string>> CCSVModule::readCSVFile(const std::string& filePath)
{
	std::vector<std::vector<std::string>> vecCsvData;
	std::ifstream file(filePath);

	if (file.is_open()) {
		std::string line = "";
		// 파일 끝까지 한줄씩 읽음
		while (std::getline(file, line)) {
			std::vector<std::string> row;
			std::stringstream ss(line);
			std::string cell = "";
			while (std::getline(ss, cell, ',')) {
				row.push_back(cell);
			}
			vecCsvData.push_back(std::move(row));
		}
		file.close();
	}

	return vecCsvData;
}
