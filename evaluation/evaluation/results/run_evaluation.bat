@echo off
REM SaklAI Evaluation Launcher for Windows
REM This script helps run the evaluation suite easily

echo.
echo ===============================================
echo       SaklAI Evaluation Suite Launcher
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "evaluation_script.py" (
    echo ERROR: evaluation_script.py not found
    echo Please run this script from the evaluation directory
    pause
    exit /b 1
)

REM Check if .venv is activated (optional check)
if not defined VIRTUAL_ENV (
    echo WARNING: Virtual environment not detected
    echo Consider activating your .venv before running
    echo.
)

REM Menu selection
echo Select evaluation mode:
echo 1. Functional Tests (Basic functionality)
echo 2. Performance Tests (Load testing)
echo 3. Comprehensive Tests (Extended test cases)
echo 4. All Tests (Complete evaluation suite)
echo 5. Quick Test (Fast functional test)
echo.
set /p choice="Enter your choice (1-5): "

REM Set default parameters
set duration=60
set users=5
set mode=functional

REM Process choice
if "%choice%"=="1" (
    set mode=functional
    echo Running Functional Tests...
) else if "%choice%"=="2" (
    set mode=performance
    echo Running Performance Tests...
    set /p duration="Test duration in seconds [60]: " || set duration=60
    set /p users="Concurrent users [5]: " || set users=5
) else if "%choice%"=="3" (
    set mode=comprehensive
    echo Running Comprehensive Tests...
) else if "%choice%"=="4" (
    set mode=all
    echo Running All Tests...
    set /p duration="Performance test duration in seconds [60]: " || set duration=60
    set /p users="Concurrent users [5]: " || set users=5
) else if "%choice%"=="5" (
    set mode=functional
    echo Running Quick Test...
    python evaluation_script.py --output results\quick_test.json
    echo.
    echo Quick test completed! Check results\quick_test.json
    pause
    exit /b 0
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

echo.
echo Starting evaluation with the following parameters:
echo Mode: %mode%
echo Duration: %duration%s
echo Concurrent Users: %users%
echo.

REM Run the evaluation
python run_evaluation.py --mode %mode% --duration %duration% --concurrent-users %users%

REM Check exit code
if errorlevel 1 (
    echo.
    echo ===============================================
    echo         EVALUATION FAILED
    echo ===============================================
    echo Check the error messages above for details
) else (
    echo.
    echo ===============================================
    echo         EVALUATION COMPLETED SUCCESSFULLY
    echo ===============================================
    echo Check the results directory for detailed reports
)

echo.
echo Press any key to exit...
pause >nul
