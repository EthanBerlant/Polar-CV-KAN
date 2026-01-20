# Overnight Benchmark Suite for CV-KAN
# Runs the Champion configurations for Image, Audio, and Timeseries

Write-Host "Starting Overnight Benchmark Suite..." -ForegroundColor Green
Write-Host "This will run the Champion configurations derived from previous optimization steps."

$StartTime = Get-Date

# Run the benchmark runner in Champion mode
# Using python directly
python experiments/run_benchmark.py --champion --resume

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "Benchmark Suite Complete!" -ForegroundColor Green
Write-Host "Total Duration: $Duration"
Write-Host "Results saved to outputs/benchmark"
