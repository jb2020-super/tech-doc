#include <iostream>
#include <codecvt>
/*
*	convert byte array to utf-16 string
*/
std::u16string byte_array_2_u16(const char* byte, int len, bool is_big_endian) {
	std::u16string u16;
	char code[2]{};
	int idx{ 0 };
	while (idx + 1 < len) {
		if (is_big_endian){
			code[1] = byte[idx];
			code[0] = byte[idx + 1];
		}else{
			code[0] = byte[idx];
			code[1] = byte[idx + 1];
		}
		char16_t* c16 = (char16_t*)code;
		u16.push_back(c16[0]);
		idx += 2;
	}
	return u16;
}

int main() {
	char name[] = {
		0, 67, 
		0, 82,
		255, 35,
		255, 6,
		255, 39,
		48, 214,
		48, 252,
		48, 177
	};
	auto u16 = byte_array_2_u16(name, 16, true);

	return 0;
}