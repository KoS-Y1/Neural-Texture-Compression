module Utils;

__include "Utils/utils_brdf.slang";
__include "Utils/utils_ibl.slang";
__include "Utils/utils_ntc.slang";

public static const float PI = 3.14159265359f;
public static const float LN2 = 0.69314718056f;

public static const float4x4 kIdentity = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };