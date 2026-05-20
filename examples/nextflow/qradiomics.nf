// =============================================================================
// qradiomics.nf — generic Nextflow template that drives the v0.9 qr CLI
// =============================================================================
//
// Designed for cohorts where each patient has a DICOM CT series and an
// RTSTRUCT contour file. Each patient flows through the pipeline in
// parallel; Nextflow caches successful processes so re-runs only touch
// patients that failed or whose inputs changed.
//
//   per-patient                                          collect
//   ┌──────────┐   ┌──────────┐   ┌───────────────┐   ┌──────────┐   ┌─────────┐
//   │ convert  ├──▶│ convert  ├──▶│ extract       ├──▶│ gather   ├──▶│ analyze │
//   │ dicom-ct │   │ rtstruct │   │ (1 patient)   │   │ features │   │         │
//   └──────────┘   └──────────┘   └───────────────┘   └──────────┘   └─────────┘
//
// Run:
//   nextflow run qradiomics.nf \
//     --dataset_root /path/to/cohort \
//     --clinical    /path/to/clinical.csv \
//     --roi         GTV-1 \
//     --pattern     nsclc-survival \
//     --analysis    survival \
//     --outdir      results
//
// Requires `qradiomics[rtstruct]` on PATH (or a container image; see
// `nextflow.config` for an example container profile).
// =============================================================================

nextflow.enable.dsl = 2

// ─── Parameters ──────────────────────────────────────────────────────────────
params.dataset_root = null     // root containing per-patient directories
params.clinical     = null     // patient_id-indexed clinical CSV
params.roi          = 'GTV'    // RTSTRUCT ROI name (case-insensitive)
params.pattern      = 'ct-default'  // qr pattern_id
params.analysis     = 'survival'    // survival | classify | importance
params.time_col     = 'OS_days'
params.event_col    = 'OS_event'
params.outcome      = 'OS_event'    // for classify/importance
params.outdir       = 'results'

if (!params.dataset_root) {
    exit 1, "Missing --dataset_root (root containing per-patient directories)"
}

// ─── Channel: per-patient (id, ct_dir, rtstruct) tuples ──────────────────────
Channel
    .fromPath("${params.dataset_root}/*", type: 'dir')
    .map { pat ->
        // Heuristic: find a CT directory + an RTSTRUCT under the patient tree
        def ct = pat.list().findAll { it.toFile().isDirectory() }
                     .find { it.name =~ /(?i)CT/ } ?: pat
        def rs = pat.toFile().listFiles(true).find { f ->
            f.name =~ /^(RS|RTSTRUCT)\..*\.dcm?$/
        }
        if (ct == null || rs == null) return null
        tuple(pat.name, ct, file(rs.absolutePath))
    }
    .filter { it != null }
    .set { patient_inputs }

// ─── 1. DICOM CT → NRRD ──────────────────────────────────────────────────────
process convert_ct {
    tag "$pid"
    publishDir "${params.outdir}/nrrd", mode: 'copy'

    input:
    tuple val(pid), path(ct_dir), path(rs)

    output:
    tuple val(pid), path("${pid}_CT.nrrd"), path(rs)

    script:
    """
    qr convert dicom-series -i ${ct_dir} -o ${pid}_CT.nrrd
    """
}

// ─── 2. RTSTRUCT → label NRRD ────────────────────────────────────────────────
process convert_rt {
    tag "$pid"
    publishDir "${params.outdir}/nrrd", mode: 'copy'

    input:
    tuple val(pid), path(ct_nrrd), path(rs)

    output:
    tuple val(pid), path(ct_nrrd), path("${pid}_${params.roi}-label.nrrd")

    script:
    """
    # Recover the DICOM CT directory for qr convert rtstruct
    # The structure set must match the CT it was drawn on, so we use the
    # original DICOM directory passed in via params.dataset_root.
    qr convert rtstruct \\
        -d ${params.dataset_root}/${pid} \\
        -r ${rs} \\
        --roi ${params.roi} \\
        -o ${pid}_${params.roi}-label.nrrd
    """
}

// ─── 3. Per-patient extraction ───────────────────────────────────────────────
process extract_one {
    tag "$pid"

    input:
    tuple val(pid), path(ct), path(mask)

    output:
    path "${pid}_features.csv"

    script:
    """
    cat > manifest.csv <<EOF
patient_id,modality,image_path,mask_path
${pid},CT,${ct},${mask}
EOF
    qr extract -m manifest.csv -p ${params.pattern} -o ${pid}_features.csv
    """
}

// ─── 4. Gather per-patient feature CSVs → single features.csv ────────────────
process gather_features {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path "feat_*.csv"

    output:
    path "features.csv"

    script:
    """
    # Header from the first file, then append data rows from each
    files=( feat_*.csv )
    head -1 \${files[0]} > features.csv
    for f in "\${files[@]}"; do
        tail -n +2 \$f >> features.csv
    done
    """
}

// ─── 5. Merge with clinical CSV ──────────────────────────────────────────────
process merge_clinical {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path features
    path clinical

    output:
    path "analysis_ready.csv"

    when:
    params.clinical != null

    script:
    """
    qr results merge \\
        -f ${features} \\
        -c ${clinical} \\
        --clinical-id-col patient_id \\
        --time-col ${params.time_col} \\
        --event-col ${params.event_col} \\
        -o analysis_ready.csv
    """
}

// ─── 6. Analyze ──────────────────────────────────────────────────────────────
process analyze {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path analysis_ready

    output:
    path "${params.analysis}_results.csv"

    script:
    if (params.analysis == 'survival')
        """
        qr analyze survival -i ${analysis_ready} \\
            --outcome OS_months --event OS_event \\
            -o ${params.analysis}_results.csv --top-n 20
        """
    else if (params.analysis == 'classify')
        """
        qr analyze classify -i ${analysis_ready} \\
            --outcome ${params.outcome} \\
            -o ${params.analysis}_results.csv --top-n 20
        """
    else
        """
        qr analyze importance -i ${analysis_ready} \\
            --outcome ${params.outcome} \\
            -o ${params.analysis}_results.csv --top-n 20
        """
}

// ─── Workflow ────────────────────────────────────────────────────────────────
workflow {
    patient_inputs
        | convert_ct
        | convert_rt
        | extract_one
        | collect
        | gather_features

    if (params.clinical) {
        merge_clinical(gather_features.out, file(params.clinical)) | analyze
    }
}
