param(
    [Parameter(Mandatory = $true)]
    [string]$DocxPath,

    [Parameter(Mandatory = $true)]
    [string]$PdfPath
)

$ErrorActionPreference = "Stop"

function Replace-AllText {
    param($Document, [string]$OldText, [string]$NewText)
    $range = $Document.Content
    $find = $range.Find
    $find.ClearFormatting()
    $find.Replacement.ClearFormatting()
    [void]$find.Execute($OldText, $false, $false, $false, $false, $false, $true, 1, $false, $NewText, 2)
}

$docxFullPath = [System.IO.Path]::GetFullPath($DocxPath)
$pdfFullPath = [System.IO.Path]::GetFullPath($PdfPath)
$pdfDirectory = [System.IO.Path]::GetDirectoryName($pdfFullPath)
if (-not (Test-Path -LiteralPath $pdfDirectory)) {
    New-Item -ItemType Directory -Force -Path $pdfDirectory | Out-Null
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($docxFullPath, $false, $false)

    Replace-AllText $document "CoT 相对 Direct · GSM8K / MATH-500" "CoT 提升 · GSM8K / MATH-500"
    Replace-AllText $document "SC@5 最高点估计 · GSM8K / MATH-500" "SC@5 最高点估计（双数据集）"
    Replace-AllText $document "两数据集合计题数 · 六种协议" "两数据集 · 六协议"
    Replace-AllText $document "MATH-500 中，复杂推导使 token 增长倍率超过准确率倍率" "MATH-500 本次结果中，平均推导输出更长，token 增长倍率超过准确率倍率"

    $subtitleRange = $document.Content.Duplicate
    $subtitleFind = $subtitleRange.Find
    $subtitleFind.ClearFormatting()
    if ($subtitleFind.Execute("Direct、Chain-of-Thought、Self-Consistency 与 Self-Refine 的准确率—资源权衡", $false, $false, $false, $false, $false, $true, 0)) {
        $subtitleRange.Font.Size = 13.5
    }

    $evidenceText = "证据范围：GSM8K 1,319 题 + MATH-500 500 题；单随机种子 20260730"
    $evidenceRange = $document.Content.Duplicate
    $evidenceFind = $evidenceRange.Find
    $evidenceFind.ClearFormatting()
    if ($evidenceFind.Execute($evidenceText, $false, $false, $false, $false, $false, $true, 0)) {
        $paragraphRange = $evidenceRange.Paragraphs.Item(1).Range
        $paragraphRange.Delete() | Out-Null
    }

    $tocTitleRange = $document.Content.Duplicate
    $tocTitleFind = $tocTitleRange.Find
    $tocTitleFind.ClearFormatting()
    if ($tocTitleFind.Execute("目录", $false, $false, $false, $false, $false, $true, 0)) {
        $tocTitleParagraph = $tocTitleRange.Paragraphs.Item(1).Range
        $betweenStart = $document.Tables.Item(1).Range.End
        $betweenEnd = $tocTitleParagraph.Start
        if ($betweenEnd -gt $betweenStart) {
            $document.Range($betweenStart, $betweenEnd).Delete() | Out-Null
        }
        $tocTitleRange = $document.Content.Duplicate
        $tocTitleFind = $tocTitleRange.Find
        $tocTitleFind.ClearFormatting()
        if ($tocTitleFind.Execute("目录", $false, $false, $false, $false, $false, $true, 0)) {
            $tocTitleRange.Paragraphs.Item(1).Range.ParagraphFormat.PageBreakBefore = -1
        }
    }

    foreach ($table in $document.Tables) {
        if ($table.Range.Text -like "*结论先行*") {
            $table.Rows.AllowBreakAcrossPages = 0
        }
        if ($table.Columns.Count -eq 2 -and $table.Range.Text -like "*本报告用途*") {
            $table.AllowAutoFit = $false
            $table.Columns.Item(1).PreferredWidthType = 3
            $table.Columns.Item(1).PreferredWidth = 205
            $table.Columns.Item(2).PreferredWidthType = 3
            $table.Columns.Item(2).PreferredWidth = 263
            $table.Columns.Item(1).Width = 205
            $table.Columns.Item(2).Width = 263
        }
    }

    foreach ($shape in $document.InlineShapes) {
        $altText = $shape.AlternativeText
        if ($altText -match '^图 2\s') {
            $shape.LockAspectRatio = -1
            $shape.Width = 410
        }
        elseif ($altText -match 'Pareto') {
            $shape.LockAspectRatio = -1
            $shape.Width = 400
        }
    }

    $pathMarkers = @("configs/", "results/", "environment/", "scripts/", "src/", "multiseed_matrix.yaml")
    foreach ($paragraph in $document.Paragraphs) {
        $paragraphText = $paragraph.Range.Text
        foreach ($marker in $pathMarkers) {
            if ($paragraphText.Contains($marker)) {
                $paragraph.Range.ParagraphFormat.Alignment = 0
                break
            }
        }
    }

    foreach ($toc in $document.TablesOfContents) {
        $toc.Update()
        $toc.UpdatePageNumbers()
    }
    $document.Repaginate()
    $document.Save()

    if (Test-Path -LiteralPath $pdfFullPath) {
        Remove-Item -LiteralPath $pdfFullPath -Force
    }
    $document.ExportAsFixedFormat($pdfFullPath, 17)
}
finally {
    if ($null -ne $document) {
        $document.Close($false)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output $docxFullPath
Write-Output $pdfFullPath
