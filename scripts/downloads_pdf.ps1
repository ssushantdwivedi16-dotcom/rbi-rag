# Downloads RBI Commercial Bank Directions into data\raw_pdfs
New-Item -ItemType Directory -Force -Path "data\raw_pdfs" | Out-Null

$ids = (150..167) + (169..172) + 174
foreach ($i in $ids) {
  Invoke-WebRequest "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/${i}MD.PDF" -OutFile "data\raw_pdfs\${i}MD.pdf"
  Write-Host "Downloaded $i"
}