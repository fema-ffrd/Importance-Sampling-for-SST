param(
  [Parameter(Mandatory)][string]$InputPath,
  [Parameter(Mandatory)][string]$OutputDir,
  [string]$SIPath,
  [string]$Image = "goconsequence:mbi-v1"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "Docker is not installed or not on PATH."
}

$InputPath = (Resolve-Path $InputPath).Path
$OutputDir = (Resolve-Path (New-Item -ItemType Directory -Force -Path $OutputDir)).Path

# SI mapping
$siMount = @()
$siInContainer = "/app/structure_inventory/si.shp"
if ($SIPath) {
  $SIPath = (Resolve-Path $SIPath).Path
  if (-not (Test-Path $SIPath -PathType Leaf)) { throw "SI shapefile not found: $SIPath" }
  $siDir  = Split-Path -Parent $SIPath
  $siBase = Split-Path -Leaf $SIPath
  $siMount = @("--mount", "type=bind,src=$siDir,dst=/si,readonly")
  $siInContainer = "/si/$siBase"
}

function Run-One([string]$tif) {
  if (-not (Test-Path $tif -PathType Leaf)) { return }
  $name = [IO.Path]::GetFileNameWithoutExtension($tif)

  docker run --rm `
    --mount "type=bind,src=$tif,dst=/in/input.tif,readonly" `
    --mount "type=bind,src=$OutputDir,dst=/out" `
    $siMount `
    $Image `
    -raster "/in/input.tif" `
    -si "$siInContainer" `
    -result "/out/${name}_result.shp"
}

if (Test-Path $InputPath -PathType Leaf) {
  if ($InputPath -notmatch '\.tif$|\.TIF$') { throw "Input file must be .tif" }
  Run-One $InputPath
} elseif (Test-Path $InputPath -PathType Container) {
  $tifs = Get-ChildItem $InputPath -Filter *.tif -File
  if (-not $tifs) { $tifs = Get-ChildItem $InputPath -Filter *.TIF -File }
  if (-not $tifs) { throw "No .tif files found in $InputPath" }
  $tifs | ForEach-Object { Run-One $_.FullName }
} else {
  throw "InputPath must be a file or directory"
}

"Done."