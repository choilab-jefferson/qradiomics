"""
PatternLoader: Loads and validates normalized pattern templates.

Loads pattern templates from backend/patterns/templates/ and provides:
- Pattern validation against schema
- Pattern merging with overrides
- Pattern compatibility checking with datasets

Architecture:
- Separate from PatternDiscoveryService (which handles legacy patterns)
- Returns Pydantic models for type safety
- Caches loaded patterns in memory
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for Pattern Structure
# =============================================================================


class ModalityRequirement(BaseModel):
    """Single modality requirement."""

    modality: str
    required: bool = True
    description: Optional[str] = None
    time_points: Optional[List[str]] = None


class ClinicalVariable(BaseModel):
    """Clinical variable definition."""

    name: str
    type: str = "any"
    description: Optional[str] = None


class PatternRequirements(BaseModel):
    """Pattern data requirements."""

    modalities: List[ModalityRequirement] = Field(default_factory=list)
    anatomical_region: Optional[str] = None
    roi_types: List[str] = Field(default_factory=list)
    clinical_variables: Optional[Dict[str, List[ClinicalVariable]]] = None
    min_subjects: int = 10
    time_points: Optional[Dict[str, Any]] = None


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    crop: Optional[Dict[str, Any]] = None
    resample: Optional[Dict[str, Any]] = None
    normalization: Optional[Dict[str, Any]] = None
    intensity_windowing: Optional[Dict[str, Any]] = None
    registration: Optional[Dict[str, Any]] = None


class DeltaFeatureConfig(BaseModel):
    """Delta feature calculation config."""

    enabled: bool = False
    time_points: List[str] = Field(default_factory=list)
    methods: List[Any] = Field(default_factory=lambda: ["difference", "ratio"])
    include_original: bool = True
    exclude_shape: bool = True


class FeatureExtractionConfig(BaseModel):
    """Feature extraction configuration."""

    extractor: str = "pyradiomics"
    # Path to extractor config; either a single file or a per-modality mapping
    # like {"CT": "pyradiomics/ct_default.yaml"}.
    config_ref: Optional[Union[str, Dict[str, str]]] = None
    image_types: List[str] = Field(default_factory=lambda: ["Original"])
    feature_classes: List[str] = Field(
        default_factory=lambda: ["firstorder", "shape", "glcm", "glrlm", "glszm"]
    )
    settings: Optional[Dict[str, Any]] = None
    delta_features: Optional[DeltaFeatureConfig] = None
    quality_control: Optional[Dict[str, Any]] = None
    multi_roi: Optional[Dict[str, Any]] = None


class FeatureSelectionConfig(BaseModel):
    """Feature selection configuration."""

    method: str = "mrmr"
    max_features: int = 20
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.9


class ModelingConfig(BaseModel):
    """Modeling configuration."""

    algorithms: List[str] = Field(default_factory=lambda: ["RandomForest"])
    survival_algorithms: Optional[List[str]] = None
    default_params: Optional[Dict[str, Dict[str, Any]]] = None


class ValidationConfig(BaseModel):
    """Validation configuration."""

    method: str = "cross_validation"
    folds: int = 5
    stratify_by: Optional[str] = None
    repeat: int = 1
    test_size: float = 0.2


class AnalysisConfig(BaseModel):
    """Analysis configuration."""

    task: str = "classification"
    feature_selection: Optional[FeatureSelectionConfig] = None
    modeling: Optional[ModelingConfig] = None
    validation: Optional[ValidationConfig] = None
    metrics: Optional[Dict[str, Any]] = None


class ReportingConfig(BaseModel):
    """Reporting configuration."""

    include: List[str] = Field(default_factory=list)
    compliance_checks: List[str] = Field(default_factory=list)
    format: str = "html"


class ExecutionConfig(BaseModel):
    """Execution configuration."""

    parallelization: Optional[Dict[str, Any]] = None
    caching: Optional[Dict[str, Any]] = None
    nextflow: Optional[Dict[str, Any]] = None


class SnippetReference(BaseModel):
    """Reference to a COMB snippet."""

    name: str = Field(..., description="Snippet name in COMB (e.g., 'icc_filter')")
    version: str = Field(default="latest", description="Version ('latest', 'v1', 'v2', etc.)")
    required: bool = Field(default=True, description="Whether snippet is required for execution")
    description: Optional[str] = Field(None, description="Role of this snippet in the pattern")

    @property
    def ref(self) -> str:
        """Get snippet reference string (e.g., 'icc_filter:v1')."""
        return f"{self.name}:{self.version}"


class PatternTemplate(BaseModel):
    """Complete pattern template model."""

    pattern_id: str
    name: str
    version: str = "1.0"
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # COMB snippet references
    snippets: List[SnippetReference] = Field(
        default_factory=list,
        description="References to reusable code snippets stored in COMB",
    )

    requirements: PatternRequirements = Field(default_factory=PatternRequirements)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    feature_extraction: FeatureExtractionConfig = Field(default_factory=FeatureExtractionConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


# =============================================================================
# PatternLoader Service
# =============================================================================


class PatternLoader:
    """
    Loads and manages normalized pattern templates.

    Responsibilities:
    - Load patterns from templates directory
    - Validate patterns against schema
    - Cache patterns in memory
    - Provide pattern lookup and search
    """

    def __init__(self, patterns_dir: Optional[Path] = None):
        """
        Initialize PatternLoader.

        Args:
            patterns_dir: Root directory for patterns (defaults to qradiomics/data)
        """
        if patterns_dir is None:
            # Determine patterns directory relative to this file
            self_path = Path(__file__).parent  # qradiomics/
            patterns_dir = self_path / "data"

        self.patterns_dir = patterns_dir
        self.templates_dir = patterns_dir / "templates"
        self.pyradiomics_dir = patterns_dir / "pyradiomics"
        self.schema_dir = patterns_dir / "schema"

        # Cache loaded patterns
        self._pattern_cache: Dict[str, PatternTemplate] = {}
        self._patterns_loaded = False

    def load_all_patterns(self) -> Dict[str, PatternTemplate]:
        """
        Load all pattern templates from templates directory.

        Returns:
            Dict mapping pattern_id to PatternTemplate
        """
        if self._patterns_loaded:
            return self._pattern_cache

        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return {}

        yaml_files = list(self.templates_dir.glob("*.yaml")) + list(
            self.templates_dir.glob("*.yml")
        )

        for yaml_file in yaml_files:
            try:
                pattern = self._load_pattern_file(yaml_file)
                if pattern:
                    self._pattern_cache[pattern.pattern_id] = pattern
                    logger.info(f"Loaded pattern: {pattern.pattern_id} ({pattern.name})")
            except Exception as e:
                logger.error(f"Failed to load pattern from {yaml_file}: {e}")

        self._patterns_loaded = True
        logger.info(f"Loaded {len(self._pattern_cache)} pattern templates")

        return self._pattern_cache

    def get_pattern(self, pattern_id: str) -> Optional[PatternTemplate]:
        """
        Get a pattern by ID.

        Args:
            pattern_id: Pattern identifier (e.g., "delta-radiomics")

        Returns:
            PatternTemplate or None if not found
        """
        if not self._patterns_loaded:
            self.load_all_patterns()

        return self._pattern_cache.get(pattern_id)

    def list_patterns(self) -> List[Dict[str, Any]]:
        """
        List all available patterns with summary info.

        Returns:
            List of pattern summary dictionaries
        """
        if not self._patterns_loaded:
            self.load_all_patterns()

        return [
            {
                "pattern_id": p.pattern_id,
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "tags": p.tags,
                "task": p.analysis.task,
                "modalities": [m.modality for m in p.requirements.modalities],
                "anatomical_region": p.requirements.anatomical_region,
                "snippets": [s.ref for s in p.snippets],
                "snippet_count": len(p.snippets),
            }
            for p in self._pattern_cache.values()
        ]

    def find_compatible_patterns(
        self,
        modalities: List[str],
        anatomical_region: Optional[str] = None,
        task: Optional[str] = None,
    ) -> List[PatternTemplate]:
        """
        Find patterns compatible with given data characteristics.

        Args:
            modalities: List of available modalities (e.g., ["CT", "PT"])
            anatomical_region: Anatomical region (e.g., "heart")
            task: Analysis task (e.g., "classification", "survival")

        Returns:
            List of compatible PatternTemplates, sorted by compatibility score
        """
        if not self._patterns_loaded:
            self.load_all_patterns()

        compatible = []

        for pattern in self._pattern_cache.values():
            score = self._compute_compatibility_score(pattern, modalities, anatomical_region, task)
            if score > 0:
                compatible.append((score, pattern))

        # Sort by score descending
        compatible.sort(key=lambda x: x[0], reverse=True)

        return [p for _, p in compatible]

    def get_pyradiomics_config(self, config_ref: str) -> Optional[Dict[str, Any]]:
        """
        Load a PyRadiomics configuration file.

        Args:
            config_ref: Reference path (e.g., "pyradiomics/ct_default.yaml")

        Returns:
            PyRadiomics config dictionary or None
        """
        # Handle both relative and absolute refs
        if config_ref.startswith("pyradiomics/"):
            config_path = self.patterns_dir / config_ref
        else:
            config_path = self.pyradiomics_dir / config_ref

        if not config_path.exists():
            logger.warning(f"PyRadiomics config not found: {config_path}")
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load PyRadiomics config: {e}")
            return None

    def merge_pattern_with_overrides(
        self, pattern_id: str, overrides: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Merge pattern template with user overrides.

        Args:
            pattern_id: Pattern identifier
            overrides: Dictionary of override values

        Returns:
            Merged configuration dictionary
        """
        pattern = self.get_pattern(pattern_id)
        if not pattern:
            return None

        # Convert pattern to dict
        pattern_dict = pattern.model_dump()

        # Deep merge overrides
        merged = self._deep_merge(pattern_dict, overrides)

        return merged

    def _load_pattern_file(self, yaml_path: Path) -> Optional[PatternTemplate]:
        """
        Load and validate a single pattern file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            PatternTemplate or None on failure
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        # Validate required fields
        if "pattern_id" not in data:
            logger.warning(f"Pattern missing pattern_id: {yaml_path}")
            return None

        if "name" not in data:
            logger.warning(f"Pattern missing name: {yaml_path}")
            return None

        # Parse into Pydantic model
        return PatternTemplate.model_validate(data)

    def _compute_compatibility_score(
        self,
        pattern: PatternTemplate,
        modalities: List[str],
        anatomical_region: Optional[str],
        task: Optional[str],
    ) -> float:
        """
        Compute compatibility score between pattern and data characteristics.

        Returns score from 0.0 (incompatible) to 1.0 (perfect match).
        """
        score = 0.0
        max_score = 0.0

        # Check required modalities
        for req in pattern.requirements.modalities:
            if req.required:
                max_score += 1.0
                if req.modality in modalities:
                    score += 1.0
                else:
                    # Required modality missing = incompatible
                    return 0.0
            else:
                max_score += 0.5
                if req.modality in modalities:
                    score += 0.5

        # Check anatomical region
        if pattern.requirements.anatomical_region:
            max_score += 1.0
            if anatomical_region:
                if pattern.requirements.anatomical_region == anatomical_region:
                    score += 1.0
                elif pattern.requirements.anatomical_region == "general":
                    score += 0.5

        # Check task
        if task:
            max_score += 1.0
            if pattern.analysis.task == task:
                score += 1.0

        if max_score == 0:
            return 0.5  # No requirements = somewhat compatible

        return score / max_score

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Override values take precedence over base values.
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def reload_patterns(self) -> None:
        """Force reload all patterns from disk."""
        self._pattern_cache.clear()
        self._patterns_loaded = False
        self.load_all_patterns()


# =============================================================================
# COMB Integration Service (Read-Only Catalog)
# =============================================================================


class SnippetCatalog:
    """
    Read-only catalog of COMB snippets for QRadiomics.

    QRadiomics uses this to:
    - List available snippets for pattern composition
    - Get snippet metadata and code for preview
    - Validate pattern snippet references

    Actual code generation/combination is delegated to HIVE Builder.
    """

    def __init__(self, project: Optional[str] = None):
        """
        Initialize SnippetCatalog.

        Args:
            project: Optional HIVE project name for project-scoped snippets.
                     None uses global COMB (~/.comb/)
        """
        self._comb: Any = None
        self._project = project
        self._comb_checked = False

    @property
    def comb(self) -> Any:
        """COMB integration archived - patterns loaded from local YAML only."""
        if not self._comb_checked:
            self._comb_checked = True
            # HIVE COMB removed - see backend/archive/hive/
            logger.debug("COMB disabled - using local patterns only")
            self._comb = None
        return self._comb

    def is_available(self) -> bool:
        """Check if COMB is available."""
        return self.comb is not None

    def list_snippets(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available snippets from COMB.

        Args:
            tag: Optional tag filter (e.g., "radiomics", "qc")

        Returns:
            List of snippet info dicts with name, tags, latest version, and extended metadata
        """
        if not self.is_available():
            return []

        snippets = []
        for name in self.comb.list(type="snippet", tag=tag):
            meta = self.comb.get_meta(name, type="snippet")
            if meta:
                snippets.append(
                    {
                        "name": meta.name,
                        "tags": meta.tags,
                        "latest_version": meta.latest,
                        "versions": meta.versions,
                        "created": meta.created.isoformat() if meta.created else None,
                        "updated": meta.updated.isoformat() if meta.updated else None,
                        # Extended metadata fields
                        "description": meta.description,
                        "category": meta.category,
                        "dependencies": meta.dependencies,
                        "requirements": meta.requirements,
                    }
                )
        return snippets

    def get_snippet_meta(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific snippet.

        Args:
            name: Snippet name

        Returns:
            Metadata dict or None if not found
        """
        if not self.is_available():
            return None

        meta = self.comb.get_meta(name, type="snippet")
        if not meta:
            return None

        return {
            "name": meta.name,
            "tags": meta.tags,
            "latest_version": meta.latest,
            "versions": meta.versions,
            "created": meta.created.isoformat() if meta.created else None,
            "updated": meta.updated.isoformat() if meta.updated else None,
            "provenance": meta.provenance.to_dict() if meta.provenance else None,
            # Extended metadata fields
            "description": meta.description,
            "category": meta.category,
            "dependencies": meta.dependencies,
            "requirements": meta.requirements,
            "inputs": [io.to_dict() for io in meta.inputs] if meta.inputs else [],
            "outputs": [io.to_dict() for io in meta.outputs] if meta.outputs else [],
            "usage_example": meta.usage_example,
        }

    def get_snippet_code(self, name: str, version: str = "latest") -> Optional[str]:
        """
        Get snippet code for preview.

        Args:
            name: Snippet name
            version: Version to retrieve ("latest" or "v1", "v2", etc.)

        Returns:
            Code string or None if not found
        """
        if not self.is_available():
            return None

        try:
            return self.comb.retrieve(name, version=version, type="snippet")
        except FileNotFoundError:
            return None

    def search_snippets(self, query: str) -> List[Dict[str, Any]]:
        """
        Search snippets by name and tags.

        Args:
            query: Search query

        Returns:
            List of (snippet_info, relevance_score) tuples
        """
        if not self.is_available():
            return []

        results = []
        for name, score in self.comb.search(query, type="snippet"):
            meta = self.comb.get_meta(name, type="snippet")
            if meta:
                results.append(
                    {
                        "name": meta.name,
                        "tags": meta.tags,
                        "latest_version": meta.latest,
                        "relevance_score": score,
                    }
                )
        return results

    def validate_pattern_snippets(self, pattern: "PatternTemplate") -> Dict[str, Any]:
        """
        Validate that all snippets referenced by a pattern exist.

        Args:
            pattern: PatternTemplate to validate

        Returns:
            Validation result with status, missing snippets, etc.
        """
        if not self.is_available():
            return {
                "valid": False,
                "error": "COMB not available",
                "missing": [],
                "found": [],
            }

        missing = []
        found = []

        for snippet_ref in pattern.snippets:
            meta = self.comb.get_meta(snippet_ref.name, type="snippet")
            if meta is None:
                missing.append(snippet_ref.name)
            elif snippet_ref.version != "latest" and snippet_ref.version not in meta.versions:
                missing.append(f"{snippet_ref.name}:{snippet_ref.version}")
            else:
                found.append(snippet_ref.name)

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "found": found,
            "total_required": len([s for s in pattern.snippets if s.required]),
        }

    def get_snippet_with_dependencies(
        self, name: str, version: str = "latest"
    ) -> Optional[Dict[str, Any]]:
        """
        Get snippet code and all its dependencies.

        Args:
            name: Snippet name
            version: Version to retrieve

        Returns:
            Dict with snippet code, dependencies code, and all requirements
        """
        if not self.is_available():
            return None

        try:
            # Use COMB's retrieve_with_dependencies
            code_dict = self.comb.retrieve_with_dependencies(name, version=version, type="snippet")
            if not code_dict:
                return None

            # Get all requirements
            all_reqs = self.comb.get_all_requirements(name, type="snippet")

            # Get dependency tree
            deps = self.comb.get_dependencies(name, type="snippet", recursive=True)

            return {
                "name": name,
                "version": version,
                "code": code_dict,
                "dependencies": deps,
                "all_requirements": all_reqs,
            }
        except FileNotFoundError:
            return None

    def get_snippet_requirements(self, name: str) -> List[str]:
        """
        Get all pip requirements for a snippet including its dependencies.

        Args:
            name: Snippet name

        Returns:
            List of pip requirements (e.g., ["pandas>=1.0", "numpy"])
        """
        if not self.is_available():
            return []

        return self.comb.get_all_requirements(name, type="snippet")


class PatternInstantiator:
    """
    Instantiates patterns by delegating to HIVE Builder.

    QRadiomics defines patterns (what to build), HIVE Builder builds them (how to build).
    This class handles the handoff between the two systems.
    """

    def __init__(self):
        self._hive_client = None

    def create_instantiation_request(
        self,
        pattern: PatternTemplate,
        dataset_config: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a pattern instantiation request for HIVE.

        This prepares the request that will be sent to HIVE Builder
        to generate executable code from the pattern.

        Args:
            pattern: PatternTemplate to instantiate
            dataset_config: Dataset-specific configuration (paths, subjects, etc.)
            overrides: Optional pattern setting overrides

        Returns:
            Instantiation request dict for HIVE
        """
        # Collect snippet references
        snippet_refs = [
            {
                "name": s.name,
                "version": s.version,
                "required": s.required,
                "description": s.description,
            }
            for s in pattern.snippets
        ]

        # Build the request
        request = {
            "type": "pattern_instantiation",
            "pattern": {
                "id": pattern.pattern_id,
                "name": pattern.name,
                "version": pattern.version,
                "task": pattern.analysis.task,
            },
            "snippets": snippet_refs,
            "config": {
                "requirements": pattern.requirements.model_dump(),
                "preprocessing": pattern.preprocessing.model_dump(),
                "feature_extraction": pattern.feature_extraction.model_dump(),
                "analysis": pattern.analysis.model_dump(),
                "reporting": pattern.reporting.model_dump(),
                "execution": pattern.execution.model_dump(),
            },
            "dataset": dataset_config,
            "overrides": overrides or {},
        }

        return request

    def get_instantiation_prompt(
        self,
        pattern: PatternTemplate,
        dataset_config: Dict[str, Any],
    ) -> str:
        """
        Generate a natural language prompt for HIVE Builder.

        This creates a prompt that HIVE's Builder agent can understand
        and use to generate the appropriate code.

        Args:
            pattern: PatternTemplate to instantiate
            dataset_config: Dataset-specific configuration

        Returns:
            Natural language prompt string
        """
        snippet_list = ", ".join([s.name for s in pattern.snippets if s.required])

        prompt = f"""Generate a {pattern.analysis.task} pipeline for {pattern.name}.

Pattern: {pattern.pattern_id} (v{pattern.version})
Description: {pattern.description}

Required Snippets from COMB:
{snippet_list}

Dataset Configuration:
- Data path: {dataset_config.get('data_path', 'N/A')}
- Modalities: {dataset_config.get('modalities', [])}
- Subjects: {dataset_config.get('subject_count', 'N/A')} subjects

Please:
1. Retrieve the required snippets from COMB
2. Adapt them to the dataset configuration
3. Combine into an executable pipeline
4. Include proper error handling and logging
"""
        return prompt


# =============================================================================
# Singleton instances for convenience
# =============================================================================

_pattern_loader: Optional[PatternLoader] = None
_snippet_catalog: Optional[SnippetCatalog] = None
_pattern_instantiator: Optional[PatternInstantiator] = None


def get_pattern_loader() -> PatternLoader:
    """Get singleton PatternLoader instance."""
    global _pattern_loader
    if _pattern_loader is None:
        _pattern_loader = PatternLoader()
    return _pattern_loader


def get_snippet_catalog(project: Optional[str] = None) -> SnippetCatalog:
    """Get singleton SnippetCatalog instance."""
    global _snippet_catalog
    if _snippet_catalog is None:
        _snippet_catalog = SnippetCatalog(project=project)
    return _snippet_catalog


def get_pattern_instantiator() -> PatternInstantiator:
    """Get singleton PatternInstantiator instance."""
    global _pattern_instantiator
    if _pattern_instantiator is None:
        _pattern_instantiator = PatternInstantiator()
    return _pattern_instantiator
