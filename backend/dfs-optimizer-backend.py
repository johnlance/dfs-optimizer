import os
import io
import csv
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DFS Optimizer API",
    description="Daily Fantasy Sports lineup optimizer supporting NFL main slate and showdown formats",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
DEFAULT_SALARY_CAP = 50000

# Valid NFL positions
VALID_POSITIONS = {"QB", "RB", "WR", "TE", "DST", "DEF", "D"}

# Position mappings for flexibility
POSITION_MAPPINGS = {
    "D": "DST",
    "DEF": "DST",
    "DEFENSE": "DST"
}

# ======================
# Pydantic Models
# ======================

class Player(BaseModel):
    name: str
    position: str
    salary: int
    projection: float
    team: str
    dk_id: Optional[str] = None
    roster_position: Optional[str] = None  # CPT or FLEX for showdown
    game_info: Optional[str] = None
    captain_salary: Optional[int] = None  # For showdown captain pricing
    
    @validator('position')
    def validate_position(cls, v):
        v = v.upper()
        # Map alternative position names
        v = POSITION_MAPPINGS.get(v, v)
        if v not in VALID_POSITIONS:
            raise ValueError(f"Invalid position: {v}")
        return v
    
    @validator('salary')
    def validate_salary(cls, v):
        if v < 0 or v > 20000:
            raise ValueError(f"Invalid salary: {v}")
        return v
    
    @validator('projection')
    def validate_projection(cls, v):
        if v < 0 or v > 100:
            raise ValueError(f"Invalid projection: {v}")
        return v

class OptimizationRequest(BaseModel):
    format_type: str = Field(..., description="'main' or 'showdown'")
    salary_cap: int = DEFAULT_SALARY_CAP
    players: List[Player]
    
    @validator('format_type')
    def validate_format_type(cls, v):
        if v not in ["main", "showdown"]:
            raise ValueError("format_type must be 'main' or 'showdown'")
        return v

class LineupResult(BaseModel):
    players: List[Player]
    total_salary: int
    total_projection: float
    remaining_salary: int
    is_valid: bool
    captain_player: Optional[Player] = None  # For showdown only

class OptimizationResponse(BaseModel):
    success: bool
    lineup: Optional[LineupResult] = None
    error_message: Optional[str] = None
    warnings: List[str] = []

class FormatInfo(BaseModel):
    name: str
    description: str
    positions: Dict[str, Any]
    salary_cap: int
    total_players: int

# ======================
# CSV Processing Module
# ======================

def parse_dk_salary_csv(file_content: bytes) -> Tuple[List[Player], List[str]]:
    """Parse DraftKings salary export format"""
    warnings = []
    players = []
    
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                content = file_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Unable to decode CSV file")
        
        # Use pandas for robust CSV parsing
        df = pd.read_csv(io.StringIO(content))
        
        # Try to identify DraftKings format by looking for expected columns
        expected_cols = ['Position', 'Name', 'ID', 'Roster Position', 'Salary', 'TeamAbbrev']
        
        # Check if we have the expected columns (case-insensitive)
        df_cols_lower = [col.lower() for col in df.columns]
        
        # Find column indices
        col_mapping = {}
        for expected in expected_cols:
            for idx, col in enumerate(df.columns):
                if expected.lower() in col.lower():
                    col_mapping[expected] = col
                    break
        
        if len(col_mapping) < 4:  # Need at least position, name, salary, team
            # Try to detect if columns are offset (first 7 columns for lineup construction)
            if len(df.columns) >= 16:
                # Standard DK format with offset
                col_mapping = {
                    'Position': df.columns[7] if len(df.columns) > 7 else None,
                    'Name': df.columns[9] if len(df.columns) > 9 else None,
                    'ID': df.columns[10] if len(df.columns) > 10 else None,
                    'Roster Position': df.columns[11] if len(df.columns) > 11 else None,
                    'Salary': df.columns[12] if len(df.columns) > 12 else None,
                    'TeamAbbrev': df.columns[14] if len(df.columns) > 14 else None,
                    'AvgPointsPerGame': df.columns[15] if len(df.columns) > 15 else None
                }
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Skip empty rows or instruction rows
                if pd.isna(row.get(col_mapping.get('Name', 'Name'), '')):
                    continue
                
                name = str(row.get(col_mapping.get('Name', 'Name'), '')).strip()
                if not name or name.lower() in ['name', 'player', '']:
                    continue
                
                position = str(row.get(col_mapping.get('Position', 'Position'), '')).strip().upper()
                position = POSITION_MAPPINGS.get(position, position)
                
                if position not in VALID_POSITIONS:
                    warnings.append(f"Skipping player {name} with invalid position: {position}")
                    continue
                
                # Parse salary
                salary_str = str(row.get(col_mapping.get('Salary', 'Salary'), '0'))
                salary = int(float(salary_str.replace('$', '').replace(',', '')))
                
                # Get team
                team = str(row.get(col_mapping.get('TeamAbbrev', 'TeamAbbrev'), 'UNK')).strip()
                
                # Get projection (use AvgPointsPerGame if available)
                projection = 0.0
                if 'AvgPointsPerGame' in col_mapping and col_mapping['AvgPointsPerGame']:
                    try:
                        projection = float(row.get(col_mapping['AvgPointsPerGame'], 0))
                    except:
                        projection = 0.0
                
                # Get DK ID
                dk_id = None
                if 'ID' in col_mapping and col_mapping['ID']:
                    dk_id = str(row.get(col_mapping['ID'], ''))
                
                # Get roster position (CPT/FLEX for showdown)
                roster_position = None
                captain_salary = None
                if 'Roster Position' in col_mapping and col_mapping['Roster Position']:
                    roster_position = str(row.get(col_mapping['Roster Position'], '')).strip()
                    if roster_position == 'CPT':
                        captain_salary = salary
                
                # Get game info
                game_info = None
                if 'Game Info' in df.columns:
                    game_info = str(row.get('Game Info', ''))
                
                player = Player(
                    name=name,
                    position=position,
                    salary=salary,
                    projection=projection,
                    team=team,
                    dk_id=dk_id,
                    roster_position=roster_position,
                    game_info=game_info,
                    captain_salary=captain_salary
                )
                
                # Handle showdown format - create both CPT and FLEX versions
                if roster_position:
                    players.append(player)
                else:
                    # For non-showdown, just add the player
                    players.append(player)
                    
            except Exception as e:
                warnings.append(f"Error parsing row {idx}: {str(e)}")
                continue
        
        if not players:
            raise ValueError("No valid players found in CSV")
            
        warnings.append(f"Successfully parsed {len(players)} players")
        
    except Exception as e:
        raise ValueError(f"Error parsing DraftKings CSV: {str(e)}")
    
    return players, warnings

def parse_custom_csv(file_content: bytes) -> Tuple[List[Player], List[str]]:
    """Parse user-uploaded CSV with flexible column detection"""
    warnings = []
    players = []
    
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                content = file_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Unable to decode CSV file")
        
        df = pd.read_csv(io.StringIO(content))
        
        # Flexible column detection
        col_mapping = {}
        
        # Position column
        for col in df.columns:
            if any(term in col.lower() for term in ['position', 'pos']):
                col_mapping['position'] = col
                break
        
        # Name column
        for col in df.columns:
            if any(term in col.lower() for term in ['name', 'player']):
                col_mapping['name'] = col
                break
        
        # Salary column
        for col in df.columns:
            if any(term in col.lower() for term in ['salary', 'sal', 'cost']):
                col_mapping['salary'] = col
                break
        
        # Projection column
        for col in df.columns:
            if any(term in col.lower() for term in ['projection', 'proj', 'points', 'fpts', 'fantasy']):
                col_mapping['projection'] = col
                break
        
        # Team column
        for col in df.columns:
            if any(term in col.lower() for term in ['team', 'tm']):
                col_mapping['team'] = col
                break
        
        # Validate we have minimum required columns
        if not all(k in col_mapping for k in ['name', 'position', 'salary']):
            missing = [k for k in ['name', 'position', 'salary'] if k not in col_mapping]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Process rows
        for idx, row in df.iterrows():
            try:
                name = str(row[col_mapping['name']]).strip()
                if not name or pd.isna(name):
                    continue
                
                position = str(row[col_mapping['position']]).strip().upper()
                position = POSITION_MAPPINGS.get(position, position)
                
                if position not in VALID_POSITIONS:
                    warnings.append(f"Skipping player {name} with invalid position: {position}")
                    continue
                
                salary = int(float(str(row[col_mapping['salary']]).replace('$', '').replace(',', '')))
                
                projection = 0.0
                if 'projection' in col_mapping:
                    try:
                        projection = float(row[col_mapping['projection']])
                    except:
                        projection = 0.0
                
                team = 'UNK'
                if 'team' in col_mapping:
                    team = str(row[col_mapping['team']]).strip()
                
                player = Player(
                    name=name,
                    position=position,
                    salary=salary,
                    projection=projection,
                    team=team
                )
                players.append(player)
                
            except Exception as e:
                warnings.append(f"Error parsing row {idx}: {str(e)}")
                continue
        
        if not players:
            raise ValueError("No valid players found in CSV")
        
        warnings.append(f"Successfully parsed {len(players)} players")
        
    except Exception as e:
        raise ValueError(f"Error parsing custom CSV: {str(e)}")
    
    return players, warnings

def parse_csv_file(file_content: bytes) -> Tuple[List[Player], List[str]]:
    """Auto-detect format and parse accordingly"""
    # Try DraftKings format first
    try:
        players, warnings = parse_dk_salary_csv(file_content)
        if players:
            return players, warnings
    except:
        pass
    
    # Fall back to custom format
    return parse_custom_csv(file_content)

# ======================
# Optimization Engine
# ======================

def optimize_main_slate(players: List[Player], salary_cap: int = 50000) -> LineupResult:
    """Optimize for main slate format"""
    # Create the LP problem
    prob = LpProblem("DFS_Main_Slate", LpMaximize)
    
    # Decision variables
    player_vars = {}
    for i, player in enumerate(players):
        player_vars[i] = LpVariable(f"player_{i}", cat='Binary')
    
    # Objective function: maximize total projection
    prob += lpSum([players[i].projection * player_vars[i] for i in range(len(players))])
    
    # Salary constraint
    prob += lpSum([players[i].salary * player_vars[i] for i in range(len(players))]) <= salary_cap
    
    # Position constraints
    position_groups = {}
    for i, player in enumerate(players):
        pos = player.position
        if pos not in position_groups:
            position_groups[pos] = []
        position_groups[pos].append(i)
    
    # QB constraint: exactly 1
    if 'QB' in position_groups:
        prob += lpSum([player_vars[i] for i in position_groups['QB']]) == 1
    else:
        raise ValueError("No QB available")
    
    # RB constraint: exactly 2
    if 'RB' in position_groups:
        prob += lpSum([player_vars[i] for i in position_groups['RB']]) >= 2
    else:
        raise ValueError("Not enough RB available")
    
    # WR constraint: exactly 3
    if 'WR' in position_groups:
        prob += lpSum([player_vars[i] for i in position_groups['WR']]) >= 3
    else:
        raise ValueError("Not enough WR available")
    
    # TE constraint: exactly 1
    if 'TE' in position_groups:
        prob += lpSum([player_vars[i] for i in position_groups['TE']]) >= 1
    else:
        raise ValueError("No TE available")
    
    # DST constraint: exactly 1
    if 'DST' in position_groups:
        prob += lpSum([player_vars[i] for i in position_groups['DST']]) == 1
    else:
        raise ValueError("No DST available")
    
    # FLEX constraint: 1 additional RB/WR/TE
    flex_indices = []
    for pos in ['RB', 'WR', 'TE']:
        if pos in position_groups:
            flex_indices.extend(position_groups[pos])
    
    # Total RB/WR/TE should be 2+3+1+1 = 7
    prob += lpSum([player_vars[i] for i in flex_indices]) == 7
    
    # Total players constraint
    prob += lpSum([player_vars[i] for i in range(len(players))]) == 9
    
    # Solve the problem
    prob.solve()
    
    if LpStatus[prob.status] != 'Optimal':
        raise ValueError("No feasible solution found")
    
    # Extract selected players
    selected_players = []
    total_salary = 0
    total_projection = 0
    
    for i in range(len(players)):
        if value(player_vars[i]) == 1:
            selected_players.append(players[i])
            total_salary += players[i].salary
            total_projection += players[i].projection
    
    # Sort by position for better display
    position_order = ['QB', 'RB', 'WR', 'TE', 'DST']
    selected_players.sort(key=lambda p: position_order.index(p.position) if p.position in position_order else 99)
    
    return LineupResult(
        players=selected_players,
        total_salary=total_salary,
        total_projection=round(total_projection, 2),
        remaining_salary=salary_cap - total_salary,
        is_valid=True
    )

def optimize_showdown(players: List[Player], salary_cap: int = 50000) -> LineupResult:
    """Optimize for showdown format"""
    # Create the LP problem
    prob = LpProblem("DFS_Showdown", LpMaximize)
    
    # Decision variables for captain and flex
    captain_vars = {}
    flex_vars = {}
    
    for i, player in enumerate(players):
        captain_vars[i] = LpVariable(f"captain_{i}", cat='Binary')
        flex_vars[i] = LpVariable(f"flex_{i}", cat='Binary')
    
    # Objective function: maximize total projection (captain gets 1.5x)
    prob += lpSum([
        players[i].projection * 1.5 * captain_vars[i] + 
        players[i].projection * flex_vars[i]
        for i in range(len(players))
    ])
    
    # Salary constraint
    # If player has captain_salary, use it; otherwise use 1.5x regular salary
    captain_salaries = []
    for i, player in enumerate(players):
        if player.captain_salary:
            captain_salaries.append(player.captain_salary)
        else:
            captain_salaries.append(int(player.salary * 1.5))
    
    prob += lpSum([
        captain_salaries[i] * captain_vars[i] + 
        players[i].salary * flex_vars[i]
        for i in range(len(players))
    ]) <= salary_cap
    
    # Exactly 1 captain
    prob += lpSum([captain_vars[i] for i in range(len(players))]) == 1
    
    # Exactly 5 flex
    prob += lpSum([flex_vars[i] for i in range(len(players))]) == 5
    
    # Player can't be both captain and flex
    for i in range(len(players)):
        prob += captain_vars[i] + flex_vars[i] <= 1
    
    # Total 6 players
    prob += lpSum([captain_vars[i] + flex_vars[i] for i in range(len(players))]) == 6
    
    # Solve the problem
    prob.solve()
    
    if LpStatus[prob.status] != 'Optimal':
        raise ValueError("No feasible solution found for showdown")
    
    # Extract selected players
    selected_players = []
    captain_player = None
    total_salary = 0
    total_projection = 0
    
    for i in range(len(players)):
        if value(captain_vars[i]) == 1:
            captain_player = players[i]
            selected_players.append(captain_player)
            total_salary += captain_salaries[i]
            total_projection += players[i].projection * 1.5
        elif value(flex_vars[i]) == 1:
            selected_players.append(players[i])
            total_salary += players[i].salary
            total_projection += players[i].projection
    
    return LineupResult(
        players=selected_players,
        total_salary=total_salary,
        total_projection=round(total_projection, 2),
        remaining_salary=salary_cap - total_salary,
        is_valid=True,
        captain_player=captain_player
    )

# ======================
# API Endpoints
# ======================

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_lineup(
    file: UploadFile = File(...),
    format_type: str = Form(...),
    salary_cap: int = Form(50000)
):
    """
    Main optimization endpoint
    - Accepts CSV file upload
    - format_type: "main" or "showdown"
    - Returns optimized lineup or error
    """
    try:
        # Validate file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Validate format type
        if format_type not in ["main", "showdown"]:
            raise HTTPException(status_code=400, detail="Invalid format_type. Must be 'main' or 'showdown'")
        
        # Parse CSV
        try:
            players, warnings = parse_csv_file(contents)
        except ValueError as e:
            return OptimizationResponse(
                success=False,
                error_message=f"CSV parsing error: {str(e)}",
                warnings=[]
            )
        
        # Run optimization
        try:
            if format_type == "main":
                lineup = optimize_main_slate(players, salary_cap)
            else:
                lineup = optimize_showdown(players, salary_cap)
            
            return OptimizationResponse(
                success=True,
                lineup=lineup,
                warnings=warnings
            )
            
        except ValueError as e:
            return OptimizationResponse(
                success=False,
                error_message=f"Optimization error: {str(e)}",
                warnings=warnings
            )
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return OptimizationResponse(
            success=False,
            error_message=f"Unexpected error: {str(e)}",
            warnings=[]
        )

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/formats")
async def get_supported_formats():
    """Return supported DFS formats and their constraints"""
    return {
        "formats": [
            FormatInfo(
                name="main",
                description="NFL Main Slate - 9 player lineup",
                positions={
                    "QB": 1,
                    "RB": 2,
                    "WR": 3,
                    "TE": 1,
                    "FLEX": 1,  # RB/WR/TE
                    "DST": 1
                },
                salary_cap=50000,
                total_players=9
            ),
            FormatInfo(
                name="showdown",
                description="NFL Showdown - 6 player lineup from single game",
                positions={
                    "Captain": 1,  # Any position, 1.5x points
                    "FLEX": 5  # Any position
                },
                salary_cap=50000,
                total_players=6
            )
        ]
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "DFS Optimizer API",
        "version": "1.0.0",
        "description": "Daily Fantasy Sports lineup optimizer",
        "endpoints": {
            "/optimize": "POST - Optimize lineup from CSV",
            "/formats": "GET - List supported formats",
            "/health": "GET - Health check"
        }
    }

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)