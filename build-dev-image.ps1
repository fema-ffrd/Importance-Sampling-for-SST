# Copy the existing Dockerfile to a new file
Copy-Item -Path "Dockerfile" -Destination "Dockerfile_temp" -Force

# Add an empty line to the new Dockerfile
Add-Content -Path "Dockerfile_temp" -Value "`n"

# Read JSON content from "interface.json" using jq, remove newlines and carriage returns, and escape double quotes
$JSONContent = Get-Content -Raw "interface.json" | ConvertFrom-Json

# Convert the JSON object to a JSON-formatted string, escape double quotes, and trim whitespace
$EscapedJSON = ($JSONContent | ConvertTo-Json -Depth 100 -Compress) -replace '"', '\"' -replace '\s+', ' '

# Append the LABEL instruction to the new Dockerfile
Add-Content -Path "Dockerfile_temp" -Value ('LABEL interface="{0}"' -f $EscapedJSON)

# Append the LABEL docsBucketName to the new Dockerfile
Add-Content -Path "Dockerfile_temp" -Value 'LABEL docsBucketName="mbi-important-sampling-dev"'

# Append the LABEL docsBucketRegion to the new Dockerfile
Add-Content -Path "Dockerfile_temp" -Value 'LABEL docsBucketRegion="us-east-1"'

# Build Docker image from the new Dockerfile
docker build -t important-sampling:dev -f Dockerfile_temp .

# Remove the temporary Dockerfile
Remove-Item -Path "Dockerfile_temp" -Force
