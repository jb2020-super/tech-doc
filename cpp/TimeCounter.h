#pragma once


#include <Windows.h>
#include <stdio.h>

/*
* How to USE:
* TimeCounter tc;
* tc.OpenConsoleWindow();
* tc.StartCount();
* // do some work
* tc.EndCount();
* tc.StandardPrint();
*/
class TimeCounter
{
public:
	TimeCounter(){
        LARGE_INTEGER freq{};
	    QueryPerformanceFrequency(&freq);
	    m_freqInv = 1.0 / freq.QuadPart;
  	}
	inline void StartCount() { QueryPerformanceCounter(&m_startTime); }
	inline void EndCount() {
		QueryPerformanceCounter(&m_endTime);
		m_current = (m_endTime.QuadPart - m_startTime.QuadPart) * m_freqInv;
		m_totalTime += m_current;
		++m_counter;
	}
	inline double GetCurrent() { return m_current * 1000; }
	inline double GetAverage() { return m_totalTime / m_counter * 1000; }
	inline double GetFPS() { return m_counter / m_totalTime; }
	inline void SetStartTime(LARGE_INTEGER &val) { m_startTime = val; }
	inline LARGE_INTEGER GetEndTime() const { return m_endTime; }
	inline LONGLONG GetCounter() const { return m_counter; }
	inline void GetStandardTime(const double total_seconds, ULONG64 &hours, ULONG &minutes, double &seconds){
		hours = static_cast<ULONG64>(total_seconds / 3600);
		minutes = static_cast<ULONG>((total_seconds - hours * 3600) / 60);
		seconds = total_seconds - hours * 3600 - minutes * 60;
	}
	// if (StartCountEx()){
	//   ...
	//   EndCountEx();
	// }
	bool StartCountEx() {
		if (m_bFirst) {
			StartCount();
			m_bFirst = false;
			return false;
		}
		else
		{
			EndCount();
			return true;
		}
	}
	void EndCountEx() {
		m_startTime = m_endTime;
	}
	void StandardPrint(const char *info) {
		ULONG64 hours{};
		ULONG minutes{};
		double seconds{};
		GetStandardTime(m_totalTime, hours, minutes, seconds);
		printf("[%lld][%s]cur: %.2lf, avg: %.2lf, fps: %.2lf total_time: %02llu:%02u:%05.2f\n", m_counter, info, GetCurrent(), GetAverage(), GetFPS(), hours, minutes, seconds);
	}
	void OpenConsoleWindow()
	{
		AllocConsole();
		freopen("CONOUT$", "w", stdout);
		if (!::IsWindowVisible(::GetConsoleWindow()))
		{
			BOOL rst = ::ShowWindow(::GetConsoleWindow(), 5);
		}
	}
private:
	LARGE_INTEGER m_startTime{0};
	LARGE_INTEGER m_endTime{0};
	LONGLONG m_counter{0};
	double m_totalTime{0.0};
	double m_freqInv{0.0};
	double m_current{0.0};
	bool m_bFirst = true;
};

