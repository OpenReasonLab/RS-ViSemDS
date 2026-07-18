from __future__ import annotations

import hashlib
import json


# Ten short, positive, category-level descriptions are used only to build
# RemoteCLIP text prototypes. Keep this file frozen during final evaluation.
AID_DESCRIPTION_ENSEMBLES = {
    "Airport": [
        "an overhead image of airport runways and taxiways",
        "an aerial scene dominated by runways aprons and terminals",
        "a remote sensing image of airport infrastructure and aircraft parking areas",
        "an airport scene with long paved runways and connected taxiways",
        "a scene showing airport aprons hangars and terminal buildings",
        "an overhead view of aircraft runways and airport service areas",
        "a large airport layout with parallel runways and open paved surfaces",
        "an airport facility with runway strips and aircraft movement areas",
        "a remote sensing scene dominated by airport transportation infrastructure",
        "an aerial airport scene with runways taxiways aprons and aircraft",
    ],
    "BareLand": [
        "an overhead image of exposed soil and cleared land",
        "a bare ground scene with human disturbed open surfaces",
        "a remote sensing image of excavation areas and irregular bare land",
        "a scene dominated by exposed earth and sparse vegetation",
        "cleared construction land with soil dirt and rough ground texture",
        "an aerial view of non sandy bare terrain and disturbed ground",
        "a bare land scene with irregular open soil surfaces",
        "an exposed ground area shaped by clearing or construction activity",
        "a remote sensing scene of human disturbed barren land",
        "an open land surface with exposed soil and limited vegetation",
    ],
    "BaseballField": [
        "an overhead image of a baseball diamond and outfield",
        "an aerial sports scene with a fan shaped baseball field",
        "a remote sensing image of a baseball infield and green outfield",
        "a baseball venue with a diamond shaped dirt infield",
        "an athletic field organized around four baseball bases",
        "an overhead view of a baseball field with stands and fences",
        "a sports complex containing a clearly marked baseball diamond",
        "an aerial scene with a curved outfield and angular infield",
        "a baseball ground surrounded by urban or recreational facilities",
        "a remote sensing scene dominated by baseball field geometry",
    ],
    "Beach": [
        "an overhead image of a sandy shoreline beside water",
        "an aerial coastal scene with beach sand and sea",
        "a remote sensing image of a bright beach water boundary",
        "a shoreline scene dominated by sand and coastal water",
        "an elongated sandy beach following the edge of water",
        "an overhead view of waves shore and pale beach material",
        "a coastal recreation scene with sand and blue water",
        "an aerial scene showing a natural land water boundary",
        "a remote sensing scene dominated by a continuous sandy coast",
        "a beach landscape with shoreline texture and adjacent water",
    ],
    "Bridge": [
        "an overhead image of a bridge crossing water or terrain",
        "an aerial scene with a narrow elevated transportation span",
        "a remote sensing image of a roadway crossing a river",
        "a bridge structure linking two sides of a water body",
        "a linear crossing with approach roads and support spans",
        "an overhead view of a bridge over transport corridors",
        "an elongated engineered structure across a geographic barrier",
        "a transportation scene centered on a connected bridge deck",
        "a remote sensing scene with bridge piers and crossing geometry",
        "an aerial bridge scene with roads extending from both ends",
    ],
    "Center": [
        "an overhead image of a dense central urban district",
        "an aerial city center with compact blocks and major roads",
        "a remote sensing image of downtown buildings and plazas",
        "an urban core with large civic or mixed use structures",
        "a dense center scene organized around streets and open squares",
        "an overhead view of central city blocks and intersections",
        "a metropolitan center with concentrated buildings and paved space",
        "an urban landmark district with complex central organization",
        "a remote sensing scene dominated by a compact city core",
        "an aerial downtown scene with dense construction and road networks",
    ],
    "Church": [
        "an overhead image of a church building and surrounding grounds",
        "an aerial religious complex with a distinctive worship structure",
        "a remote sensing image of a church roof tower and courtyard",
        "a church scene with a prominent central building footprint",
        "a religious facility surrounded by roads vegetation or residences",
        "an overhead view of a church complex and open gathering space",
        "an architectural landmark with church like roof geometry",
        "an aerial scene centered on a worship building and grounds",
        "a remote sensing scene dominated by a church property",
        "a church complex with a distinctive main hall and auxiliary buildings",
    ],
    "Commercial": [
        "an overhead image of commercial buildings and business blocks",
        "a commercial urban area with shops offices and service facilities",
        "a business district with large buildings roads and parking areas",
        "a mixed use commercial scene with retail and office structures",
        "an aerial view of shopping centers office blocks and paved surfaces",
        "a remote sensing scene dominated by business and service facilities",
        "a commercial area with shopping buildings roads and parking lots",
        "an urban business scene with organized roads and commercial blocks",
        "a retail and office district with dense urban infrastructure",
        "a commercial land use scene with shops offices services and parking",
    ],
    "DenseResidential": [
        "an overhead image of tightly packed residential buildings",
        "a high density housing area with small gaps between roofs",
        "a dense residential scene with repeated houses and narrow roads",
        "a remote sensing image of continuous residential blocks",
        "an urban neighborhood dominated by compact housing",
        "a residential area with high building coverage and dense streets",
        "a dense housing pattern with repeated small buildings",
        "an aerial view of crowded residential blocks and local roads",
        "a residential scene with limited open space between buildings",
        "a compact neighborhood with continuous housing texture",
    ],
    "Desert": [
        "an overhead image of natural arid desert terrain",
        "an aerial scene dominated by sand and barren landforms",
        "a remote sensing image of dunes and dry natural surfaces",
        "a desert landscape with broad uniform sandy texture",
        "an arid scene with sparse vegetation and natural bare ground",
        "an overhead view of wind shaped sand patterns",
        "a dry landscape dominated by natural desert morphology",
        "an aerial desert scene with dunes ridges and open terrain",
        "a remote sensing scene of extensive sandy barren land",
        "a natural desert surface with minimal built infrastructure",
    ],
}


NWPU_DESCRIPTION_ENSEMBLES = {
    "dense_residential": [
        "an overhead image of very compact residential buildings",
        "a high density housing area with small gaps between buildings",
        "a dense residential scene with repeated roofs and narrow roads",
        "a remote sensing image of continuous residential blocks",
        "an urban neighborhood dominated by tightly packed houses",
        "a compact residential area with limited open space",
        "a dense housing pattern with repeated small buildings",
        "an aerial view of crowded residential blocks and local streets",
        "a residential scene with very high building coverage",
        "a high density neighborhood with continuous housing texture",
    ],
    "medium_residential": [
        "an overhead image of moderately spaced residential buildings",
        "a residential neighborhood with medium building density",
        "an aerial housing area with visible roads trees and open gaps",
        "a remote sensing image of organized residential blocks",
        "a neighborhood with continuous housing and moderate spacing",
        "a residential scene balancing buildings vegetation and streets",
        "an urban housing pattern with moderate roof coverage",
        "an aerial view of separated houses in organized blocks",
        "a medium density residential district with regular local roads",
        "a residential land use scene with moderate open space",
    ],
    "sparse_residential": [
        "an overhead image of scattered residential buildings",
        "a low density housing area with large gaps between houses",
        "an aerial residential scene with abundant vegetation and open land",
        "a remote sensing image of isolated houses along local roads",
        "a sparse neighborhood with low building coverage",
        "a residential scene dominated by open space around homes",
        "an aerial view of widely separated houses and vegetation",
        "a low density settlement with discontinuous residential texture",
        "a sparse housing pattern distributed across open terrain",
        "a residential land use scene with few buildings and broad spacing",
    ],
    "mobile_home_park": [
        "an overhead image of small mobile homes arranged in rows",
        "a trailer park with repeated narrow rectangular housing units",
        "an aerial scene of uniform homes organized in a regular grid",
        "a remote sensing image of parallel rows of mobile housing",
        "a compact park of similarly sized elongated residential units",
        "an overhead view of repetitive trailer like buildings and lanes",
        "a mobile home community with highly regular unit orientation",
        "an aerial housing scene defined by small repeated rectangles",
        "a residential park with dense rows of uniform prefabricated homes",
        "a remote sensing scene dominated by orderly mobile home patterns",
    ],
    "commercial_area": [
        "an overhead image of commercial buildings and business blocks",
        "a commercial urban area with offices shops and service facilities",
        "a business district with large buildings roads and parking areas",
        "a mixed use commercial scene with retail and office structures",
        "an aerial view of shopping centers office blocks and paved surfaces",
        "a remote sensing scene dominated by business and service facilities",
        "a commercial area with shopping buildings roads and parking lots",
        "an urban business scene with organized roads and commercial blocks",
        "a retail and office district with dense urban infrastructure",
        "a commercial land use scene with shops offices services and parking",
    ],
    "industrial_area": [
        "an overhead image of factories warehouses and storage yards",
        "an industrial district with large rectangular production buildings",
        "an aerial scene of warehouse roofs service roads and open yards",
        "a remote sensing image of manufacturing and logistics facilities",
        "an industrial complex with tanks containers and factory structures",
        "an overhead view of large sheds and organized storage space",
        "a production area dominated by broad roofs and utility yards",
        "an aerial industrial layout with freight access and service roads",
        "a remote sensing scene of factories warehouses and logistics land",
        "an industrial land use scene with large facilities and paved yards",
    ],
    "parking_lot": [
        "an overhead image of marked parking spaces and vehicles",
        "a large paved parking area organized into vehicle rows",
        "an aerial scene dominated by parked cars and driving aisles",
        "a remote sensing image of a broad open parking surface",
        "a parking facility with regular striping and repeated vehicles",
        "an overhead view of parking bays access lanes and cars",
        "an urban paved scene dominated by vehicle storage",
        "an aerial parking lot with geometric rows and marked spaces",
        "a remote sensing scene centered on an extensive parking area",
        "a paved land use scene with dense or sparse parked vehicles",
    ],
    "railway_station": [
        "an overhead image of railway tracks platforms and station buildings",
        "an aerial transport scene with multiple parallel rail lines",
        "a railway station layout with platforms tracks and access roads",
        "a remote sensing image of rail corridors and station facilities",
        "a station scene organized around long linear railway structures",
        "an overhead view of platforms trains and parallel tracks",
        "a rail transport facility with switching lines and service buildings",
        "an aerial railway scene with elongated platforms and track geometry",
        "a remote sensing scene dominated by station related rail infrastructure",
        "a railway land use scene with tracks platforms and terminal structures",
    ],
}


AID_BOUNDARY_RULES = {
    "Airport": "Choose Airport when runways, taxiways, aprons, terminals, hangars, or aircraft form a dominant airport-specific layout; large paved areas alone are insufficient.",
    "BareLand": "Choose BareLand for exposed soil, clearing, excavation, construction disturbance, or irregular non-sandy ground; natural dunes and broad arid landforms indicate Desert.",
    "BaseballField": "Choose BaseballField when a diamond-shaped infield and curved outfield are visible; a generic stadium, track, or plaza without baseball geometry is insufficient.",
    "Beach": "Choose Beach when a sandy shoreline and clear land-water boundary dominate the scene.",
    "Bridge": "Choose Bridge when a narrow engineered span clearly crosses water, roads, railways, or a terrain gap and connects transportation approaches.",
    "Center": "Choose Center for a compact central urban core with civic or landmark organization, plazas, major intersections, and dense mixed structures; retail buildings and parking dominance indicate Commercial.",
    "Church": "Choose Church when a distinctive worship-building footprint, tower, nave-like roof, or religious complex is the central evidence; an isolated large roof alone is insufficient.",
    "Commercial": "Choose Commercial for retail, office, service, or mixed-use blocks with access roads and parking; repeated compact housing indicates DenseResidential and civic-core organization indicates Center.",
    "DenseResidential": "Choose DenseResidential when tightly packed repeated housing, small gaps, and dense local roads dominate; large retail buildings and parking indicate Commercial.",
    "Desert": "Choose Desert for natural arid terrain, dunes, broad sandy texture, and sparse vegetation; artificial clearing, excavation, or construction disturbance indicates BareLand.",
}


NWPU_BOUNDARY_RULES = {
    "dense_residential": "Choose dense_residential for compact continuous housing with very high roof coverage, small gaps, repeated residential roofs, and limited open space.",
    "medium_residential": "Choose medium_residential for organized continuous housing with moderate density, visible streets, vegetation, and gaps larger than dense residential but smaller than sparse residential.",
    "sparse_residential": "Choose sparse_residential for scattered houses, low building coverage, wide spacing, and abundant vegetation or open land.",
    "mobile_home_park": "Choose mobile_home_park for many small uniform trailer-like rectangular units arranged in highly regular rows or grids.",
    "commercial_area": "Choose commercial_area for business, shopping, office, service, retail, or mixed-use blocks; factories, warehouses, storage yards, tanks, containers, or logistics facilities indicate industrial_area.",
    "industrial_area": "Choose industrial_area when factories, warehouses, large production roofs, storage yards, tanks, containers, freight access, or logistics facilities dominate.",
    "parking_lot": "Choose parking_lot only when marked parking spaces, parked vehicles, and open paved parking surfaces dominate; nearby buildings should remain secondary.",
    "railway_station": "Choose railway_station when parallel railway tracks, platforms, station buildings, trains, or rail-corridor geometry are clearly visible and dominant.",
}


ALIASES = {
    "aid": "aid",
    "nwpu": "nwpu_fg_urban",
    "nwpu_urban": "nwpu_fg_urban",
    "nwpu-urban": "nwpu_fg_urban",
    "nwpu_fg_urban": "nwpu_fg_urban",
}


def canonical_dataset_name(dataset: str) -> str:
    value = ALIASES.get(dataset.strip().lower())
    if value is None:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return value


def description_ensembles(dataset: str, class_order: list[str]) -> dict[str, list[str]]:
    canonical = canonical_dataset_name(dataset)
    source = AID_DESCRIPTION_ENSEMBLES if canonical == "aid" else NWPU_DESCRIPTION_ENSEMBLES
    _validate_text_map(source, class_order, expected_count=10)
    return {label: list(source[label]) for label in class_order}


def boundary_rules(dataset: str, class_order: list[str]) -> dict[str, str]:
    canonical = canonical_dataset_name(dataset)
    source = AID_BOUNDARY_RULES if canonical == "aid" else NWPU_BOUNDARY_RULES
    _validate_text_map(source, class_order)
    return {label: source[label] for label in class_order}


def category_text_sha256(dataset: str, class_order: list[str]) -> str:
    payload = {
        "descriptions": description_ensembles(dataset, class_order),
        "boundary_rules": boundary_rules(dataset, class_order),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_text_map(source: dict, class_order: list[str], expected_count: int | None = None) -> None:
    missing = [label for label in class_order if label not in source]
    if missing:
        raise ValueError(f"Missing category texts for: {missing}")
    if expected_count is not None:
        wrong = {label: len(source[label]) for label in class_order if len(source[label]) != expected_count}
        if wrong:
            raise ValueError(f"Each category requires {expected_count} descriptions: {wrong}")

