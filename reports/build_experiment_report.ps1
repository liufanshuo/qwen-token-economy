param(
    [Parameter(Mandatory = $true)]
    [string]$MarkdownPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string]$QaPdfPath = ""
)

$ErrorActionPreference = "Stop"

function Get-WordColor {
    param([string]$Hex)
    $clean = $Hex.TrimStart('#')
    $r = [Convert]::ToInt32($clean.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($clean.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($clean.Substring(4, 2), 16)
    return $r + ($g * 256) + ($b * 65536)
}

function Set-StyleFont {
    param(
        $Style,
        [string]$LatinName,
        [string]$EastAsiaName,
        [double]$Size,
        [string]$Color,
        [bool]$Bold = $false,
        [bool]$Italic = $false
    )
    $Style.Font.Name = $LatinName
    $Style.Font.NameFarEast = $EastAsiaName
    $Style.Font.Size = $Size
    $Style.Font.Color = Get-WordColor $Color
    $Style.Font.Bold = if ($Bold) { -1 } else { 0 }
    $Style.Font.Italic = if ($Italic) { -1 } else { 0 }
}

function Set-CellShading {
    param($Cell, [string]$Hex)
    $Cell.Shading.BackgroundPatternColor = Get-WordColor $Hex
}

function Clean-InlineMarkup {
    param([string]$Text)
    return $Text.Replace('**', '').Replace('`', '')
}

function Add-PlainParagraph {
    param(
        $Selection,
        [string]$Text,
        $Style,
        [int]$Alignment = -1,
        [bool]$KeepWithNext = $false
    )
    $Selection.Style = $Style
    if ($Alignment -ge 0) {
        $Selection.ParagraphFormat.Alignment = $Alignment
    }
    $Selection.ParagraphFormat.KeepWithNext = if ($KeepWithNext) { -1 } else { 0 }
    $Selection.TypeText((Clean-InlineMarkup $Text))
    $Selection.TypeParagraph()
}

function Add-LeadCallout {
    param($Document, $Selection, [string]$Label, [string]$Text)
    $table = $Document.Tables.Add($Selection.Range, 1, 1)
    $table.AllowAutoFit = $false
    $table.PreferredWidthType = 3
    $table.PreferredWidth = 468
    $table.Rows.Alignment = 0
    $table.Rows.AllowBreakAcrossPages = 0
    $table.Rows.LeftIndent = 6
    $table.TopPadding = 6
    $table.BottomPadding = 6
    $table.LeftPadding = 8
    $table.RightPadding = 8
    Set-CellShading $table.Cell(1, 1) "F4F6F9"
    $table.Borders.Enable = 0
    $cellRange = $table.Cell(1, 1).Range
    $cellRange.End = $cellRange.End - 1
    $cellRange.Text = "$Label  $Text"
    $cellRange.Style = $Document.Styles.Item(-1)
    $cellRange.Font.Name = "Calibri"
    $cellRange.Font.NameFarEast = "Microsoft YaHei"
    $cellRange.Font.Size = 10.5
    $cellRange.Font.Color = Get-WordColor "0B2545"
    $cellRange.ParagraphFormat.Alignment = 0
    $cellRange.ParagraphFormat.FirstLineIndent = 0
    $cellRange.ParagraphFormat.SpaceAfter = 0
    $cellRange.ParagraphFormat.LineSpacingRule = 5
    $cellRange.ParagraphFormat.LineSpacing = 15
    $labelRange = $table.Cell(1, 1).Range.Duplicate
    $labelRange.End = [Math]::Min($labelRange.Start + $Label.Length, $labelRange.End - 1)
    $labelRange.Bold = -1
    $Selection.SetRange($table.Range.End, $table.Range.End)
    $Selection.TypeParagraph()
}

function Add-CodeBlock {
    param($Document, $Selection, [string]$Text)
    $table = $Document.Tables.Add($Selection.Range, 1, 1)
    $table.AllowAutoFit = $false
    $table.PreferredWidthType = 3
    $table.PreferredWidth = 468
    $table.Rows.Alignment = 0
    $table.Rows.LeftIndent = 6
    $table.TopPadding = 5
    $table.BottomPadding = 5
    $table.LeftPadding = 8
    $table.RightPadding = 8
    Set-CellShading $table.Cell(1, 1) "F2F4F7"
    $table.Borders.Enable = 0
    $range = $table.Cell(1, 1).Range
    $range.End = $range.End - 1
    $range.Text = $Text
    $range.Style = $Document.Styles.Item(-1)
    $range.Font.Name = "Consolas"
    $range.Font.NameFarEast = "Microsoft YaHei"
    $range.Font.Size = 8.8
    $range.Font.Color = Get-WordColor "30363D"
    $range.ParagraphFormat.Alignment = 0
    $range.ParagraphFormat.FirstLineIndent = 0
    $range.ParagraphFormat.SpaceAfter = 0
    $Selection.SetRange($table.Range.End, $table.Range.End)
    $Selection.TypeParagraph()
}

function Get-ColumnWidths {
    param([int]$ColumnCount)
    switch ($ColumnCount) {
        2 { return @(115, 353) }
        3 { return @(105, 248, 115) }
        4 { return @(78, 250, 78, 62) }
        5 { return @(88, 92, 112, 82, 94) }
        6 { return @(68, 78, 86, 75, 82, 79) }
        7 { return @(76, 72, 103, 62, 64, 45, 46) }
        8 { return @(62, 48, 102, 66, 60, 56, 35, 39) }
        default {
            $width = [Math]::Floor(468 / $ColumnCount)
            $values = @()
            for ($i = 0; $i -lt $ColumnCount; $i++) { $values += $width }
            $values[$ColumnCount - 1] += 468 - (($ColumnCount - 1) * $width + $width)
            return $values
        }
    }
}

function Add-MarkdownTable {
    param($Document, $Selection, [string[]]$TableLines)
    $rows = @()
    foreach ($line in $TableLines) {
        $trimmed = $line.Trim().Trim('|')
        $cells = @($trimmed.Split('|') | ForEach-Object { (Clean-InlineMarkup $_.Trim()) })
        $rows += ,$cells
    }
    if ($rows.Count -lt 2) { return }
    $separator = $rows[1]
    $dataRows = @()
    $dataRows += ,$rows[0]
    if (($separator | Where-Object { $_ -notmatch '^:?-{3,}:?$' }).Count -eq 0) {
        if ($rows.Count -gt 2) {
            for ($rowIndex = 2; $rowIndex -lt $rows.Count; $rowIndex++) {
                $dataRows += ,$rows[$rowIndex]
            }
        }
    }
    else {
        $dataRows = @()
        foreach ($row in $rows) { $dataRows += ,$row }
    }

    $columnCount = $dataRows[0].Count
    $table = $Document.Tables.Add($Selection.Range, $dataRows.Count, $columnCount)
    $table.AllowAutoFit = $false
    $table.PreferredWidthType = 3
    $table.PreferredWidth = 468
    $table.Rows.Alignment = 0
    $table.Rows.LeftIndent = 6
    $table.TopPadding = 4
    $table.BottomPadding = 4
    $table.LeftPadding = 6
    $table.RightPadding = 6
    $table.Spacing = 0
    $table.Rows.AllowBreakAcrossPages = 0
    $table.Borders.Enable = 1
    $table.Borders.OutsideColor = Get-WordColor "D9DEE5"
    $table.Borders.InsideColor = Get-WordColor "D9DEE5"
    $table.Rows.Item(1).HeadingFormat = -1

    $headerText = $dataRows[0] -join '|'
    if ($columnCount -eq 4 -and $headerText -match '部署目标') {
        $widths = @(90, 70, 155, 153)
    }
    elseif ($columnCount -eq 2 -and $headerText -match '本报告用途') {
        $widths = @(205, 263)
    }
    elseif ($columnCount -eq 4 -and $headerText -match '跨数据集解释') {
        $widths = @(92, 92, 92, 192)
    }
    elseif ($columnCount -eq 4 -and $headerText -match '配对差值') {
        $widths = @(78, 200, 110, 80)
    }
    else {
        $widths = Get-ColumnWidths $columnCount
    }
    for ($c = 1; $c -le $columnCount; $c++) {
        $table.Columns.Item($c).PreferredWidthType = 3
        $table.Columns.Item($c).PreferredWidth = $widths[$c - 1]
    }

    for ($r = 1; $r -le $dataRows.Count; $r++) {
        for ($c = 1; $c -le $columnCount; $c++) {
            $cell = $table.Cell($r, $c)
            $cell.VerticalAlignment = 1
            $range = $cell.Range
            $range.End = $range.End - 1
            $range.Text = [string]$dataRows[$r - 1][$c - 1]
            $range.Style = $Document.Styles.Item(-1)
            $range.ParagraphFormat.OutlineLevel = 10
            $range.Font.Name = "Calibri"
            $range.Font.NameFarEast = "Microsoft YaHei"
            $range.Font.Size = if ($columnCount -ge 7) { 7.6 } elseif ($columnCount -ge 5) { 8.3 } else { 9.0 }
            $range.ParagraphFormat.FirstLineIndent = 0
            $range.ParagraphFormat.SpaceBefore = 0
            $range.ParagraphFormat.SpaceAfter = 0
            $range.ParagraphFormat.LineSpacingRule = 0
            $isNumeric = $range.Text -match '^[-+]?([0-9]|\.|,|%|×|/|–|\s)+$'
            $range.ParagraphFormat.Alignment = if ($isNumeric -or $c -gt 1 -and $columnCount -ge 5) { 1 } else { 0 }
            if ($r -eq 1) {
                Set-CellShading $cell "F4F6F9"
                $range.Bold = -1
                $range.Font.Color = Get-WordColor "1F4D78"
                $range.ParagraphFormat.Alignment = 1
            }
        }
    }
    $table.Range.ParagraphFormat.KeepWithNext = 0
    $Selection.SetRange($table.Range.End, $table.Range.End)
    $Selection.TypeParagraph()
}

function Add-CoverMetricStrip {
    param($Document, $Selection)
    $Selection.TypeParagraph()
    $Selection.TypeParagraph()
    $table = $Document.Tables.Add($Selection.Range, 1, 3)
    $table.AllowAutoFit = $false
    $table.PreferredWidthType = 3
    $table.PreferredWidth = 468
    $table.Rows.Alignment = 0
    $table.Rows.LeftIndent = 0
    $table.TopPadding = 10
    $table.BottomPadding = 10
    $table.LeftPadding = 7
    $table.RightPadding = 7
    $table.Borders.Enable = 0
    $metrics = @(
        [PSCustomObject]@{ Value = "+66.79 / +44.20 pp"; Label = "CoT 提升 · GSM8K / MATH-500" },
        [PSCustomObject]@{ Value = "92.72% / 78.00%"; Label = "SC@5 最高点估计（双数据集）" },
        [PSCustomObject]@{ Value = "1,819"; Label = "两数据集 · 六协议" }
    )
    for ($c = 1; $c -le 3; $c++) {
        $cell = $table.Cell(1, $c)
        $cell.Width = 156
        Set-CellShading $cell "F4F6F9"
        $range = $cell.Range
        $range.End = $range.End - 1
        $range.Text = $metrics[$c - 1].Value + "`r" + $metrics[$c - 1].Label
        $range.Style = $Document.Styles.Item(-1)
        $range.Font.Name = "Calibri"
        $range.Font.NameFarEast = "Microsoft YaHei"
        $range.Font.Color = Get-WordColor "0B2545"
        $range.ParagraphFormat.Alignment = 1
        $range.ParagraphFormat.FirstLineIndent = 0
        $range.ParagraphFormat.SpaceAfter = 0
        $first = $range.Paragraphs.Item(1).Range
        $first.Font.Size = 17
        $first.Bold = -1
        $second = $range.Paragraphs.Item(2).Range
        $second.Font.Size = 8.8
        $second.Font.Color = Get-WordColor "5F6B76"
    }
    $Selection.SetRange($table.Range.End, $table.Range.End)
}

$markdownFullPath = (Resolve-Path -LiteralPath $MarkdownPath).Path
$outputFullPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputPath))
$outputDirectory = [System.IO.Path]::GetDirectoryName($outputFullPath)
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}
$markdownDirectory = [System.IO.Path]::GetDirectoryName($markdownFullPath)
$lines = Get-Content -LiteralPath $markdownFullPath -Encoding UTF8

$word = $null
$document = $null
$phase = "initialize"
try {
    $phase = "launch Word"
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Add()

    $phase = "configure page"
    $section = $document.Sections.Item(1)
    $section.PageSetup.PageWidth = 612
    $section.PageSetup.PageHeight = 792
    $section.PageSetup.TopMargin = 72
    $section.PageSetup.BottomMargin = 72
    $section.PageSetup.LeftMargin = 72
    $section.PageSetup.RightMargin = 72
    $section.PageSetup.HeaderDistance = 35.424
    $section.PageSetup.FooterDistance = 35.424
    $section.PageSetup.DifferentFirstPageHeaderFooter = -1

    $phase = "configure styles"
    $normal = $document.Styles.Item(-1)
    Set-StyleFont $normal "Calibri" "Microsoft YaHei" 11 "222222"
    $normal.ParagraphFormat.Alignment = 3
    $normal.ParagraphFormat.FirstLineIndent = 22
    $normal.ParagraphFormat.SpaceBefore = 0
    $normal.ParagraphFormat.SpaceAfter = 8
    $normal.ParagraphFormat.LineSpacingRule = 5
    $normal.ParagraphFormat.LineSpacing = 16

    $heading1 = $document.Styles.Item(-2)
    Set-StyleFont $heading1 "Calibri" "Microsoft YaHei" 16 "2E74B5" $true
    $heading1.ParagraphFormat.Alignment = 0
    $heading1.ParagraphFormat.FirstLineIndent = 0
    $heading1.ParagraphFormat.SpaceBefore = 18
    $heading1.ParagraphFormat.SpaceAfter = 10
    $heading1.ParagraphFormat.KeepWithNext = -1

    $heading2 = $document.Styles.Item(-3)
    Set-StyleFont $heading2 "Calibri" "Microsoft YaHei" 13 "2E74B5" $true
    $heading2.ParagraphFormat.Alignment = 0
    $heading2.ParagraphFormat.FirstLineIndent = 0
    $heading2.ParagraphFormat.SpaceBefore = 12
    $heading2.ParagraphFormat.SpaceAfter = 6
    $heading2.ParagraphFormat.KeepWithNext = -1

    $heading3 = $document.Styles.Item(-4)
    Set-StyleFont $heading3 "Calibri" "Microsoft YaHei" 12 "1F4D78" $true
    $heading3.ParagraphFormat.Alignment = 0
    $heading3.ParagraphFormat.FirstLineIndent = 0
    $heading3.ParagraphFormat.SpaceBefore = 8
    $heading3.ParagraphFormat.SpaceAfter = 4
    $heading3.ParagraphFormat.KeepWithNext = -1

    $coverTitle = $document.Styles.Add("ReportCoverTitle", 1)
    Set-StyleFont $coverTitle "Calibri" "Microsoft YaHei" 28 "203748" $true
    $coverTitle.ParagraphFormat.Alignment = 1
    $coverTitle.ParagraphFormat.FirstLineIndent = 0
    $coverTitle.ParagraphFormat.SpaceBefore = 72
    $coverTitle.ParagraphFormat.SpaceAfter = 14
    $coverTitle.ParagraphFormat.LineSpacingRule = 0

    $coverSubtitle = $document.Styles.Add("ReportCoverSubtitle", 1)
    Set-StyleFont $coverSubtitle "Calibri" "Microsoft YaHei" 13.5 "2B5163"
    $coverSubtitle.ParagraphFormat.Alignment = 1
    $coverSubtitle.ParagraphFormat.FirstLineIndent = 0
    $coverSubtitle.ParagraphFormat.SpaceAfter = 32

    $coverMeta = $document.Styles.Add("ReportCoverMeta", 1)
    Set-StyleFont $coverMeta "Calibri" "Microsoft YaHei" 10.5 "59636E"
    $coverMeta.ParagraphFormat.Alignment = 1
    $coverMeta.ParagraphFormat.FirstLineIndent = 0
    $coverMeta.ParagraphFormat.SpaceAfter = 7

    $captionStyle = $document.Styles.Item(-35)
    Set-StyleFont $captionStyle "Calibri" "Microsoft YaHei" 9.5 "3F4C58"
    $captionStyle.ParagraphFormat.Alignment = 1
    $captionStyle.ParagraphFormat.FirstLineIndent = 0
    $captionStyle.ParagraphFormat.SpaceBefore = 5
    $captionStyle.ParagraphFormat.SpaceAfter = 7
    $captionStyle.ParagraphFormat.KeepWithNext = 0

    $evidenceStyle = $document.Styles.Add("EvidenceNote", 1)
    Set-StyleFont $evidenceStyle "Calibri" "Microsoft YaHei" 8.7 "6B7280"
    $evidenceStyle.ParagraphFormat.Alignment = 0
    $evidenceStyle.ParagraphFormat.FirstLineIndent = 0
    $evidenceStyle.ParagraphFormat.SpaceBefore = 0
    $evidenceStyle.ParagraphFormat.SpaceAfter = 7

    $tocTitle = $document.Styles.Add("ReportTOCTitle", 1)
    Set-StyleFont $tocTitle "Calibri" "Microsoft YaHei" 20 "203748" $true
    $tocTitle.ParagraphFormat.Alignment = 0
    $tocTitle.ParagraphFormat.FirstLineIndent = 0
    $tocTitle.ParagraphFormat.SpaceBefore = 20
    $tocTitle.ParagraphFormat.SpaceAfter = 14

    $phase = "configure header and footer"
    $headerRange = $section.Headers.Item(1).Range
    $headerRange.Text = "QWEN TOKEN ECONOMY  |  GSM8K + MATH-500 实验报告"
    $headerRange.Font.Name = "Calibri"
    $headerRange.Font.NameFarEast = "Microsoft YaHei"
    $headerRange.Font.Size = 8.5
    $headerRange.Font.Color = Get-WordColor "7A8793"
    $headerRange.ParagraphFormat.Alignment = 0

    $footerRange = $section.Footers.Item(1).Range
    $footerRange.Text = ""
    $footerRange.ParagraphFormat.Alignment = 2
    $footerRange.Font.Name = "Calibri"
    $footerRange.Font.Size = 8.5
    $footerRange.Font.Color = Get-WordColor "7A8793"
    $footerRange.Fields.Add($footerRange, -1, "PAGE", $true) | Out-Null

    $selection = $word.Selection
    $selection.HomeKey(6) | Out-Null
    $onCover = $true
    $coverParagraphIndex = 0
    $firstPageBreak = $true
    $listState = [PSCustomObject]@{ Start = -1; Type = "" }

    function Flush-List {
        if ($listState.Start -ge 0) {
            $range = $document.Range($listState.Start, $selection.Start)
            if ($listState.Type -eq "number") {
                $range.ListFormat.ApplyNumberDefault()
            }
            else {
                $range.ListFormat.ApplyBulletDefault()
            }
            $range.ParagraphFormat.LeftIndent = 36
            $range.ParagraphFormat.FirstLineIndent = -18
            $range.ParagraphFormat.SpaceAfter = 4
            $range.ParagraphFormat.LineSpacingRule = 5
            $range.ParagraphFormat.LineSpacing = 14.5
            $listState.Start = -1
            $listState.Type = ""
        }
    }

    $phase = "parse markdown"
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        $phase = "markdown line $($i + 1): $line"
        if ([string]::IsNullOrWhiteSpace($line)) {
            Flush-List
            continue
        }

        if ($line -eq "<!-- PAGEBREAK -->") {
            Flush-List
            if ($firstPageBreak) {
                Add-CoverMetricStrip $document $selection
                $selection.Style = $tocTitle
                $selection.ParagraphFormat.PageBreakBefore = -1
                $selection.TypeText("目录")
                $selection.TypeParagraph()
                $tocRange = $selection.Range
                $tocObject = $document.TablesOfContents.Add($tocRange, $true, 1, 3)
                $afterToc = $tocObject.Range.End
                $selection.SetRange($afterToc, $afterToc)
                $selection.InsertBreak(7)
                $firstPageBreak = $false
                $onCover = $false
            }
            else {
                $selection.InsertBreak(7)
            }
            continue
        }

        if ($line -match '^!\[(.+?)\]\((.+?)\)$') {
            Flush-List
            $caption = $matches[1]
            $relativePath = $matches[2]
            $imagePath = [System.IO.Path]::GetFullPath((Join-Path $markdownDirectory $relativePath))
            if (-not (Test-Path -LiteralPath $imagePath)) {
                throw "Image not found: $imagePath"
            }
            $selection.Style = $normal
            $selection.ParagraphFormat.Alignment = 1
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.ParagraphFormat.SpaceBefore = 5
            $selection.ParagraphFormat.SpaceAfter = 0
            $selection.ParagraphFormat.KeepWithNext = -1
            $shape = $selection.InlineShapes.AddPicture($imagePath, $false, $true)
            $shape.LockAspectRatio = -1
            $targetWidth = if ($caption -match '^图 2\s') { 410 } elseif ($caption -match 'Pareto') { 400 } else { 455 }
            if ($shape.Width -gt $targetWidth) { $shape.Width = $targetWidth }
            if ($shape.Height -gt 300) { $shape.Height = 300 }
            $shape.AlternativeText = $caption
            $selection.TypeParagraph()
            Add-PlainParagraph $selection $caption $captionStyle 1 $false
            continue
        }

        if ($line.TrimStart().StartsWith('|')) {
            Flush-List
            $tableLines = @()
            while ($i -lt $lines.Count -and $lines[$i].TrimStart().StartsWith('|')) {
                $tableLines += $lines[$i]
                $i++
            }
            $i--
            Add-MarkdownTable $document $selection $tableLines
            continue
        }

        if ($line -match '^> \[(.+?)\]\s*(.*)$') {
            Flush-List
            Add-LeadCallout $document $selection $matches[1] $matches[2]
            continue
        }

        if ($line -match '^#\s+(.+)$') {
            Flush-List
            $selection.Style = $coverMeta
            $selection.ParagraphFormat.SpaceBefore = 20
            $selection.ParagraphFormat.SpaceAfter = 10
            $selection.Font.Bold = -1
            $selection.Font.Color = Get-WordColor "A8782A"
            $selection.TypeText("EXPERIMENT REPORT  ·  2026")
            $selection.TypeParagraph()
            Add-PlainParagraph $selection $matches[1] $coverTitle 1 $false
            $coverParagraphIndex = 1
            continue
        }

        if ($line -match '^##\s+(.+)$') {
            Flush-List
            Add-PlainParagraph $selection $matches[1] $heading1 0 $true
            continue
        }

        if ($line -match '^###\s+(.+)$') {
            Flush-List
            Add-PlainParagraph $selection $matches[1] $heading2 0 $true
            continue
        }

        if ($line -match '^####\s+(.+)$') {
            Flush-List
            Add-PlainParagraph $selection $matches[1] $heading3 0 $true
            continue
        }

        if ($line -match '^\d+\.\s+(.+)$') {
            if ($listState.Start -lt 0 -or $listState.Type -ne "number") {
                Flush-List
                $listState.Start = $selection.Start
                $listState.Type = "number"
            }
            $selection.Style = $normal
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.TypeText((Clean-InlineMarkup $matches[1]))
            $selection.TypeParagraph()
            continue
        }

        if ($line -match '^[-*]\s+(.+)$') {
            if ($listState.Start -lt 0 -or $listState.Type -ne "bullet") {
                Flush-List
                $listState.Start = $selection.Start
                $listState.Type = "bullet"
            }
            $selection.Style = $normal
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.TypeText((Clean-InlineMarkup $matches[1]))
            $selection.TypeParagraph()
            continue
        }

        Flush-List
        if ($onCover) {
            if ($coverParagraphIndex -eq 1) {
                Add-PlainParagraph $selection $line $coverSubtitle 1 $false
            }
            else {
                Add-PlainParagraph $selection $line $coverMeta 1 $false
            }
            $coverParagraphIndex++
        }
        elseif ($line.StartsWith('来源：')) {
            Add-PlainParagraph $selection $line $evidenceStyle 0 $false
        }
        elseif ($line.StartsWith('`') -and $line.EndsWith('`')) {
            Add-CodeBlock $document $selection $line.Trim([char]96)
        }
        else {
            Add-PlainParagraph $selection $line $normal 3 $false
        }
    }
    Flush-List

    $phase = "update TOC and fields"
    foreach ($toc in $document.TablesOfContents) {
        $toc.Update()
    }
    $document.Fields.Update() | Out-Null

    $phase = "remove standalone blank pages"
    $pageCount = $document.ComputeStatistics(2)
    if ($pageCount -ge 5) {
        $pageNumber = 4
        $pageStart = $document.GoTo(1, 1, $pageNumber).Start
        $pageEnd = $document.GoTo(1, 1, $pageNumber + 1).Start
        $pageRange = $document.Range($pageStart, $pageEnd)
        $pageText = $pageRange.Text
        $visibleText = $pageText.Replace([string][char]12, '').Replace([string][char]13, '').Replace([string][char]7, '').Trim()
        if ($visibleText.Length -eq 0 -and $pageText.Contains([string][char]12)) {
            $pageRange.Delete() | Out-Null
        }
    }
    foreach ($toc in $document.TablesOfContents) {
        $toc.Update()
    }
    $document.Fields.Update() | Out-Null

    $phase = "set document metadata"
    try {
        $document.BuiltInDocumentProperties.Item(1).Value = "Qwen2.5-7B-Instruct 在 GSM8K 与 MATH-500 上的推理时 Token Economy 实验报告"
        $document.BuiltInDocumentProperties.Item(2).Value = "双数据集推理协议准确率与资源权衡"
        $document.BuiltInDocumentProperties.Item(3).Value = "qwen-token-economy"
        $document.BuiltInDocumentProperties.Item(5).Value = "基于仓库内配置、汇总结果、图表和环境元数据生成。"
    }
    catch {
        # Metadata is optional and may be unavailable in some localized Word builds.
    }

    $phase = "save DOCX"
    $document.SaveAs2($outputFullPath, 16)
    if ($QaPdfPath) {
        $phase = "export QA PDF"
        $qaPdfFullPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $QaPdfPath))
        $qaDirectory = [System.IO.Path]::GetDirectoryName($qaPdfFullPath)
        if (-not (Test-Path -LiteralPath $qaDirectory)) {
            New-Item -ItemType Directory -Force -Path $qaDirectory | Out-Null
        }
        $document.ExportAsFixedFormat($qaPdfFullPath, 17)
    }
}
catch {
    [Console]::Error.WriteLine("Document build failed during: $phase")
    [Console]::Error.WriteLine($_.Exception.ToString())
    [Console]::Error.WriteLine($_.ScriptStackTrace)
    throw
}
finally {
    if ($document -ne $null) {
        $document.Close($false)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($word -ne $null) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output $outputFullPath
