// Generic Nextflow pipeline behind `qr workflow run --executor nextflow`.
//
// Four static processes; every body just runs the `qr ...` command lines it
// receives as values (rendered from a WorkflowPlan by
// qradiomics.workflow.plan_to_nextflow_params), so the task hash -- and
// therefore -resume -- keys on the exact commands and the case/pair ids.
//
//   QR_PRE   once, cohort-level steps before the fan-out
//   QR_CASE  once per case id (fanout="case" steps)
//   QR_PAIR  once per (image, label) pair, joined to its case by case id
//            (fanout="pair" steps); publishes <pair>.prov.json
//   QR_POST  once, cohort-level steps after every case/pair finished
//
// params (from -params-file): plan_name, fingerprint, cwd, outdir, run_dir,
//   pre[], cases[{id,cmds[],fp,pairs[{id,cmds[],outputs[],fp}]}], post[],
//   has_pairs, on_error, max_forks, container (see nextflow.config).
// `fingerprint` (versions + container digest + spec hash [+ manifest content])
// is an input of every process. Each case/pair `fp` adds the identity it
// depends on (series_uid / rtstruct_sha).

nextflow.enable.dsl = 2

// POSIX single-quote a value for use inside a bash script body.
def q(s) { "'" + s.toString().replace("'", "'\\''") + "'" }

process QR_PRE {
    tag "${params.plan_name}"

    input:
    val cmds
    val fingerprint

    output:
    val true, emit: done

    script:
    """
    cd ${q(params.cwd)}
    ${cmds.join('\n    ')}
    """
}

process QR_CASE {
    tag { c.id }
    publishDir path: { "${params.outdir}/${c.id}" }, mode: 'copy'

    input:
    tuple val(c), val(ready), val(fingerprint)

    output:
    tuple val(c.id), val(c.fp), emit: res
    path "${c.id}.done"

    script:
    assert c.id ==~ /[A-Za-z0-9_.-]+/ : "QR_CASE: unsafe case id '${c.id}'"
    """
    echo '${c.id}' > '${c.id}.done'
    cd ${q(params.cwd)}
    ${c.cmds.join('\n    ')}
    """
}

process QR_PAIR {
    tag { p.id }
    publishDir path: { "${params.outdir}/prov" }, mode: 'copy'

    input:
    tuple val(case_id), val(p), val(ready), val(fingerprint)

    output:
    tuple val(p.id), val(p.fp), emit: res
    path "${p.id}.prov.json"

    script:
    assert p.id ==~ /[A-Za-z0-9_.-]+/ : "QR_PAIR: unsafe pair id '${p.id}'"
    """
    WD="\$PWD"
    cd ${q(params.cwd)}
    ${p.cmds.join('\n    ')}
    """
}

process QR_POST {
    tag "${params.plan_name}"

    input:
    val results
    val cmds
    val fingerprint

    output:
    val true, emit: done

    script:
    """
    cd ${q(params.cwd)}
    ${cmds.join('\n    ')}
    """
}

workflow {
    fp = Channel.value(params.fingerprint)
    QR_PRE(Channel.value(params.pre), fp)
    cases = Channel.fromList(params.cases as List).combine(QR_PRE.out.done).combine(fp)
    QR_CASE(cases)
    // sorted (id, fingerprint) pairs: QR_POST re-runs iff any case/pair's commands
    // or identity change, never because of completion order.
    if (params.has_pairs) {
        pairs = Channel.fromList(
            (params.cases as List).collectMany { c -> c.pairs.collect { p -> tuple(c.id, p) } })
        // join pairs to their case by case id: a pair starts once its case task is done
        joined = pairs.combine(QR_CASE.out.res.map { id, cfp -> tuple(id, true) }, by: 0).combine(fp)
        QR_PAIR(joined)
        results = QR_PAIR.out.res
    } else {
        results = QR_CASE.out.res
    }
    QR_POST(results.toSortedList().ifEmpty([]), Channel.value(params.post), fp)
}
