"""
Basic test script for DFS Optimizer API
"""
import requests
import json
import os

# API base URL - change this when testing deployed version
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_formats():
    """Test the formats endpoint"""
    print("Testing formats endpoint...")
    response = requests.get(f"{BASE_URL}/formats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_optimize_main_slate(csv_file_path):
    """Test main slate optimization"""
    print("Testing main slate optimization...")
    
    with open(csv_file_path, 'rb') as f:
        files = {'file': f}
        data = {
            'format_type': 'main',
            'salary_cap': 50000
        }
        response = requests.post(f"{BASE_URL}/optimize", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result['success']:
        print("Optimization successful!")
        lineup = result['lineup']
        print(f"Total Salary: ${lineup['total_salary']}")
        print(f"Total Projection: {lineup['total_projection']}")
        print(f"Remaining Salary: ${lineup['remaining_salary']}")
        print("\nLineup:")
        for player in lineup['players']:
            print(f"  {player['position']} - {player['name']} ({player['team']}) "
                  f"${player['salary']} - {player['projection']} pts")
    else:
        print(f"Optimization failed: {result['error_message']}")
    
    if result.get('warnings'):
        print(f"\nWarnings: {result['warnings']}")
    print()

def test_optimize_showdown(csv_file_path):
    """Test showdown optimization"""
    print("Testing showdown optimization...")
    
    with open(csv_file_path, 'rb') as f:
        files = {'file': f}
        data = {
            'format_type': 'showdown',
            'salary_cap': 50000
        }
        response = requests.post(f"{BASE_URL}/optimize", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result['success']:
        print("Optimization successful!")
        lineup = result['lineup']
        print(f"Total Salary: ${lineup['total_salary']}")
        print(f"Total Projection: {lineup['total_projection']}")
        print(f"Remaining Salary: ${lineup['remaining_salary']}")
        
        if lineup.get('captain_player'):
            captain = lineup['captain_player']
            print(f"\nCaptain: {captain['name']} ({captain['position']}) - "
                  f"${captain.get('captain_salary', int(captain['salary'] * 1.5))} - "
                  f"{captain['projection'] * 1.5} pts")
        
        print("\nFlex Players:")
        for player in lineup['players']:
            if player != lineup.get('captain_player'):
                print(f"  {player['position']} - {player['name']} ({player['team']}) "
                      f"${player['salary']} - {player['projection']} pts")
    else:
        print(f"Optimization failed: {result['error_message']}")
    
    if result.get('warnings'):
        print(f"\nWarnings: {result['warnings']}")
    print()

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    # Test invalid format type
    print("1. Testing invalid format type...")
    data = {
        'format_type': 'invalid',
        'salary_cap': 50000
    }
    files = {'file': ('test.csv', b'invalid,csv,data', 'text/csv')}
    response = requests.post(f"{BASE_URL}/optimize", files=files, data=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    # Test invalid CSV
    print("2. Testing invalid CSV...")
    data = {
        'format_type': 'main',
        'salary_cap': 50000
    }
    files = {'file': ('test.csv', b'this is not a valid csv', 'text/csv')}
    response = requests.post(f"{BASE_URL}/optimize", files=files, data=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

if __name__ == "__main__":
    print("DFS Optimizer API Test Suite")
    print("=" * 50)
    
    # Run basic tests
    test_health_check()
    test_formats()
    test_error_handling()
    
    # Test with sample files if they exist
    if os.path.exists("sample_custom.csv"):
        test_optimize_main_slate("sample_custom.csv")
    else:
        print("sample_custom.csv not found - skipping main slate test\n")
    
    if os.path.exists("sample_showdown.csv"):
        test_optimize_showdown("sample_showdown.csv")
    else:
        print("sample_showdown.csv not found - skipping showdown test\n")
    
    print("Test suite complete!")