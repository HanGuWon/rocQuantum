@echo off
chcp 65001
cd /d "C:\Users\한구원\Desktop\rocQuantum-1"
"C:\Program Files\CMake\bin\cmake.exe" -S "C:\Users\한구원\Desktop\rocQuantum-1" -B "C:\Users\한구원\Desktop\rocQuantum-1\build"
"C:\Program Files\CMake\bin\cmake.exe" --build "C:\Users\한구원\Desktop\rocQuantum-1\build"
