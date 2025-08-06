#ifndef CSV_MODULE_H_
#define CSV_MODULE_H_

#include <string>
#include <vector>
#include <mutex>

class CCSVModule
{
private:
	std::mutex m_mtxCSV;

public:
	CCSVModule();
	virtual ~CCSVModule();
	CCSVModule(const CCSVModule& _other) = delete;
	CCSVModule& operator=(const CCSVModule& _other) = delete;
	CCSVModule(CCSVModule&& _other) = delete;
	CCSVModule& operator=(CCSVModule&&) = delete;

	void createFolderIfNotExists(const std::string& folderPath);																		// 해당 경로에 폴더가 없으면 생성한다
	void createCSVwithHeader(const std::string& filePath, const std::vector<std::string>& header);						// 유효한 csv파일이 있는지 검사하고 없으면 지정된 헤더로 생성한다
	void appendToCSV(const std::string& filePath, const std::vector<std::string>& row);									// csv마지막 행에 데이터를 추가한다 (한개 행)
	std::vector<std::string> readCSVHeader(const std::string& filePath);														// 해당 csv파일의 첫 행 정보를 읽어 벡터로 반환
	std::vector<std::vector<std::string>> readCSVFile(const std::string& filePath);
};

#endif			// CSV_MODULE_H_
