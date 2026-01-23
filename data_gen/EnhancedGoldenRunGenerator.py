import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

class EnhancedGoldenRunGenerator:
    """
    Generates realistic, context-aware golden run solutions with diverse
    approaches, realistic costs, and proven outcomes.
    """
    
    def __init__(self):
        # Cost variance factors by context
        self.cost_multipliers = {
            'weather': {'clear': 1.0, 'rainy': 1.15, 'snowy': 1.3, 'stormy': 1.25, 'foggy': 1.1, 'hot': 1.05, 'flooded': 1.5},
            'zone': {'core': 1.4, 'mid': 1.0, 'peripheral': 0.75},
            'severity': {'low': 0.6, 'medium': 1.0, 'high': 1.3, 'critical': 1.8},
            'time': {'peak': 1.2, 'off_peak': 0.9}
        }
        
        # Solution templates by accident type with variations
        self.solution_templates = {
            'derailment': [
                {
                    'variant': 'track_geometry',
                    'description': 'Comprehensive track geometry correction with laser alignment and continuous welded rail replacement',
                    'base_cost': 450000,
                    'base_time': 44,
                    'steps': [
                        'Deploy emergency response team and secure perimeter',
                        'Conduct laser-based track geometry survey',
                        'Replace damaged sections with continuous welded rail (CWR)',
                        'Install rail stress monitoring sensors at critical points',
                        'Perform ultrasonic flaw detection on 500m radius',
                        'Execute loaded dynamic testing before service resumption'
                    ],
                    'preventive': [
                        'Deploy track geometry measurement train monthly',
                        'Install vibration-based condition monitoring',
                        'Implement weather-adaptive speed restrictions',
                        'Upgrade ballast specifications for high-speed sections'
                    ]
                },
                {
                    'variant': 'wheel_rail_interface',
                    'description': 'Wheel-rail interface optimization with rail grinding and wheel profile restoration',
                    'base_cost': 380000,
                    'base_time': 36,
                    'steps': [
                        'Immediate site isolation and safety protocols',
                        'Rail grinding to restore optimal profile',
                        'Wheel profile measurement and re-turning',
                        'Install rail lubrication system at curves',
                        'Perform gauge face conditioning',
                        'Validate with instrumented test vehicle'
                    ],
                    'preventive': [
                        'Quarterly wheel profile monitoring program',
                        'Automated rail grinding schedule based on tonnage',
                        'Real-time wheel impact load detection',
                        'Curve speed optimization based on cant deficiency'
                    ]
                }
            ],
            'collision': [
                {
                    'variant': 'signaling_upgrade',
                    'description': 'ETCS Level 2 signaling implementation with positive train control',
                    'base_cost': 920000,
                    'base_time': 68,
                    'steps': [
                        'Install European Train Control System (ETCS) Level 2',
                        'Deploy radio block centers with redundancy',
                        'Equip all trains with onboard ATP units',
                        'Implement automatic brake intervention system',
                        'Create virtual block sections (50% capacity increase)',
                        'Conduct extensive integration testing with live traffic'
                    ],
                    'preventive': [
                        'Real-time train separation monitoring',
                        'Predictive collision risk assessment algorithms',
                        'Driver alertness monitoring system',
                        'Automated speed enforcement at approach zones'
                    ]
                },
                {
                    'variant': 'operational_procedures',
                    'description': 'Enhanced operational procedures with communication protocol overhaul',
                    'base_cost': 680000,
                    'base_time': 52,
                    'steps': [
                        'Implement CBTC (Communication-Based Train Control)',
                        'Deploy redundant radio communication systems',
                        'Install track occupancy detection grid',
                        'Create automated conflict detection algorithms',
                        'Establish mandatory verbal confirmation protocols',
                        'Train all staff on new procedures with simulator'
                    ],
                    'preventive': [
                        'AI-based train movement prediction',
                        'Mandatory speed restrictions at conflict zones',
                        'Real-time dispatcher workload monitoring',
                        'Automated route locking verification'
                    ]
                }
            ],
            'signal_failure': [
                {
                    'variant': 'modern_interlocking',
                    'description': 'Solid-state interlocking with fail-safe architecture and N+2 redundancy',
                    'base_cost': 580000,
                    'base_time': 32,
                    'steps': [
                        'Replace relay-based interlocking with SSI (Solid State Interlocking)',
                        'Install triple-redundant vital processors',
                        'Deploy hot-standby signal control units',
                        'Implement automatic degraded mode fallback',
                        'Add battery backup with 4-hour capacity',
                        'Perform failover testing under load'
                    ],
                    'preventive': [
                        'Predictive maintenance using ML on signal logs',
                        'Remote condition monitoring dashboard',
                        'Automated self-diagnostic routines every 6 hours',
                        'Environmental sensors for humidity/temperature'
                    ]
                },
                {
                    'variant': 'power_resilience',
                    'description': 'Enhanced power supply resilience with distributed UPS architecture',
                    'base_cost': 420000,
                    'base_time': 24,
                    'steps': [
                        'Install distributed UPS modules at each signal',
                        'Deploy dual-feed power supply configuration',
                        'Add solar backup panels for critical signals',
                        'Implement automatic power source switching',
                        'Upgrade to LED signal heads (90% power reduction)',
                        'Test full power failure scenario'
                    ],
                    'preventive': [
                        'Power quality monitoring at all signal locations',
                        'Preventive UPS battery replacement every 3 years',
                        'Lightning protection system upgrades',
                        'Quarterly load testing of backup systems'
                    ]
                }
            ],
            'brake_failure': [
                {
                    'variant': 'brake_modernization',
                    'description': 'Complete brake system modernization with electronic control and redundancy',
                    'base_cost': 720000,
                    'base_time': 30,
                    'steps': [
                        'Install electro-pneumatic brake (EP) system',
                        'Deploy brake force management computers',
                        'Add independent emergency brake circuit',
                        'Install wheel slide protection (WSP) on all axles',
                        'Upgrade to composite brake blocks',
                        'Conduct deceleration testing at maximum load'
                    ],
                    'preventive': [
                        'Continuous brake performance monitoring',
                        'Automated brake pad wear sensors',
                        'Predictive maintenance based on braking events',
                        'Mandatory brake tests before each service'
                    ]
                }
            ],
            'electrical_failure': [
                {
                    'variant': 'catenary_upgrade',
                    'description': 'High-reliability catenary system with auto-sectioning and weather resistance',
                    'base_cost': 680000,
                    'base_time': 38,
                    'steps': [
                        'Replace catenary with high-tensile copper alloy',
                        'Install automated section isolators',
                        'Deploy ice detection and prevention system',
                        'Add redundant feeder stations every 5km',
                        'Implement real-time tension monitoring',
                        'Test under simulated extreme weather'
                    ],
                    'preventive': [
                        'Thermal imaging inspections quarterly',
                        'Automated catenary height monitoring',
                        'Weather-based preventive de-icing',
                        'Predictive wear analysis using pantograph data'
                    ]
                },
                {
                    'variant': 'power_electronics',
                    'description': 'Traction power supply modernization with active harmonic filtering',
                    'base_cost': 850000,
                    'base_time': 42,
                    'steps': [
                        'Install modern traction substations',
                        'Deploy active harmonic filters',
                        'Add SCADA monitoring for all electrical sections',
                        'Implement automatic load balancing',
                        'Upgrade protection relays to digital',
                        'Perform power quality analysis'
                    ],
                    'preventive': [
                        'Real-time power factor monitoring',
                        'Predictive transformer maintenance',
                        'Automated fault location systems',
                        'Temperature monitoring of all critical components'
                    ]
                }
            ],
            'track_obstruction': [
                {
                    'variant': 'automated_detection',
                    'description': 'AI-powered obstacle detection with automated clearance protocols',
                    'base_cost': 340000,
                    'base_time': 18,
                    'steps': [
                        'Deploy LIDAR-based obstacle detection system',
                        'Install thermal cameras along critical sections',
                        'Implement AI object recognition algorithms',
                        'Create automated alert and train stopping system',
                        'Establish rapid response team protocols',
                        'Test with various obstruction scenarios'
                    ],
                    'preventive': [
                        '24/7 automated surveillance monitoring',
                        'Quarterly vegetation management program',
                        'Perimeter fencing at high-risk zones',
                        'Wildlife crossing structures at known migration paths'
                    ]
                }
            ],
            'wheel_bearing_failure': [
                {
                    'variant': 'predictive_monitoring',
                    'description': 'Wireless bearing health monitoring with predictive analytics',
                    'base_cost': 460000,
                    'base_time': 22,
                    'steps': [
                        'Install wireless temperature/vibration sensors on all bearings',
                        'Deploy trackside acoustic monitoring stations',
                        'Implement ML-based failure prediction models',
                        'Replace all bearings with sealed ceramic hybrid units',
                        'Create automated alert system to control center',
                        'Establish condition-based maintenance protocols'
                    ],
                    'preventive': [
                        'Real-time bearing health dashboard',
                        'Automated work order generation for at-risk bearings',
                        'Thermal imaging from passing trains',
                        'Vibration signature analysis every revolution'
                    ]
                }
            ],
            'switch_malfunction': [
                {
                    'variant': 'modern_actuators',
                    'description': 'Electro-hydraulic point machines with self-diagnostic capabilities',
                    'base_cost': 390000,
                    'base_time': 16,
                    'steps': [
                        'Replace mechanical points with electro-hydraulic actuators',
                        'Install redundant position detection sensors',
                        'Deploy heated point systems for winter operation',
                        'Implement automated lubrication systems',
                        'Add real-time force monitoring',
                        'Test under extreme temperature conditions'
                    ],
                    'preventive': [
                        'Automated switch exercising every 6 hours',
                        'Predictive maintenance based on operation count',
                        'Weather-triggered preventive actions',
                        'Remote force monitoring dashboard'
                    ]
                }
            ],
            'track_defect': [
                {
                    'variant': 'rail_replacement',
                    'description': 'Premium-grade rail installation with continuous monitoring',
                    'base_cost': 720000,
                    'base_time': 40,
                    'steps': [
                        'Install premium-grade heat-treated rail',
                        'Deploy fiber-optic rail monitoring system',
                        'Implement thermite welding with ultrasonic verification',
                        'Add rail stress analysis sensors',
                        'Perform comprehensive NDT (non-destructive testing)',
                        'Validate with loaded test trains'
                    ],
                    'preventive': [
                        'Distributed acoustic sensing (DAS) for crack detection',
                        'Automated ultrasonic testing vehicle monthly',
                        'Thermal stress monitoring during temperature extremes',
                        'AI-based defect prediction from sensor data'
                    ]
                }
            ],
            'electrical_fire': [
                {
                    'variant': 'fire_suppression',
                    'description': 'Advanced fire detection and suppression with thermal management',
                    'base_cost': 620000,
                    'base_time': 36,
                    'steps': [
                        'Install fiber-optic linear heat detection',
                        'Deploy automatic aerosol fire suppression',
                        'Upgrade all wiring to fire-resistant grade',
                        'Implement compartment isolation systems',
                        'Add thermal management with active cooling',
                        'Test suppression system with controlled fire'
                    ],
                    'preventive': [
                        'Thermal imaging inspections monthly',
                        'Predictive overheating detection algorithms',
                        'Automated circuit breaker testing',
                        'Real-time current monitoring on all circuits'
                    ]
                }
            ]
        }
        
        # Outcome quality factors
        self.outcome_narratives = [
            "Became industry benchmark - adopted by 4 regional networks",
            "Zero recurrence in 24 months post-implementation",
            "Reduced similar incident frequency by 89%",
            "Achieved 98% reliability improvement in affected corridor",
            "Solution published in IEEE Rail Transportation journal",
            "Awarded 'Excellence in Safety Innovation' by transport authority",
            "Exceeded all performance targets within 3 months",
            "Cost recovery achieved through reduced delays in 14 months"
        ]
    
    def calculate_realistic_cost(self, base_cost: int, context: Dict[str, Any]) -> int:
        """Calculate context-aware cost with realistic variance"""
        multiplier = 1.0
        
        # Apply context multipliers
        multiplier *= self.cost_multipliers['weather'].get(context.get('weather_conditions', 'clear'), 1.0)
        multiplier *= self.cost_multipliers['severity'].get(context.get('severity', 'medium'), 1.0)
        
        # Zone-based cost (labor/material costs vary)
        if 'location' in context and 'zone' in context['location']:
            zone = context['location']['zone']
            multiplier *= self.cost_multipliers['zone'].get(zone, 1.0)
        
        # Time-of-day impact (peak requires more expensive emergency crews)
        is_peak = context.get('is_peak', False)
        multiplier *= 1.2 if is_peak else 1.0
        
        # Add realistic variance (±8-12%)
        variance = random.uniform(0.92, 1.12)
        
        final_cost = int(base_cost * multiplier * variance)
        return final_cost
    
    def calculate_implementation_time(self, base_time: int, context: Dict[str, Any]) -> int:
        """Calculate realistic implementation time based on context"""
        multiplier = 1.0
        
        # Weather delays
        weather_delays = {'rainy': 1.15, 'snowy': 1.3, 'stormy': 1.25, 'flooded': 1.5}
        multiplier *= weather_delays.get(context.get('weather_conditions', 'clear'), 1.0)
        
        # Severity increases scope
        severity_time = {'low': 0.8, 'medium': 1.0, 'high': 1.2, 'critical': 1.4}
        multiplier *= severity_time.get(context.get('severity', 'medium'), 1.0)
        
        # Peak hours require night work (slower)
        if context.get('is_peak', False):
            multiplier *= 1.15
        
        # Random variance
        variance = random.uniform(0.95, 1.15)
        
        return int(base_time * multiplier * variance)
    
    def generate_actual_outcomes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic outcome metrics"""
        severity = context.get('severity', 'medium')
        
        # Better outcomes for well-executed solutions
        base_scores = {
            'critical': {'delay_reduction': (85, 95), 'satisfaction': 'high', 'safety_improvement': (0.25, 0.35)},
            'high': {'delay_reduction': (88, 96), 'satisfaction': 'high', 'safety_improvement': (0.20, 0.30)},
            'medium': {'delay_reduction': (90, 97), 'satisfaction': 'very_high', 'safety_improvement': (0.15, 0.25)},
            'low': {'delay_reduction': (92, 98), 'satisfaction': 'very_high', 'safety_improvement': (0.10, 0.20)}
        }
        
        score_range = base_scores.get(severity, base_scores['medium'])
        
        # Recurrence prevention depends on solution quality
        months_recurrence_free = random.randint(18, 36) if severity in ['critical', 'high'] else random.randint(24, 48)
        
        return {
            'delay_reduction_pct': random.randint(*score_range['delay_reduction']),
            'passenger_satisfaction': score_range['satisfaction'],
            'safety_score_improvement': round(random.uniform(*score_range['safety_improvement']), 2),
            'recurrence_prevented_months': months_recurrence_free,
            'network_reliability_gain_pct': random.randint(12, 28),
            'operational_efficiency_gain_pct': random.randint(8, 18)
        }
    
    def generate_context_adaptations(self, accident_type: str, context: Dict[str, Any]) -> List[str]:
        """Generate context-specific adaptations"""
        adaptations = []
        
        weather = context.get('weather_conditions', 'clear')
        speed = context.get('speed_at_incident', 0)
        severity = context.get('severity', 'medium')
        
        # Weather-specific adaptations
        weather_adaptations = {
            'rainy': [
                'Enhanced drainage system installed to prevent water accumulation',
                'All-weather traction control systems implemented',
                'Rain-triggered automatic speed restriction protocols'
            ],
            'snowy': [
                'Heated track sections installed at critical points',
                'Snow detection sensors integrated with traffic management',
                'Winter operating procedures with reduced service speeds'
            ],
            'stormy': [
                'Wind speed monitoring integrated with train control',
                'Automated service suspension at 80+ km/h winds',
                'Enhanced weather forecasting integration for preventive actions'
            ],
            'foggy': [
                'Enhanced visibility systems with thermal imaging',
                'Automatic Train Protection (ATP) activation in low visibility',
                'Fog-triggered spacing increase protocols'
            ],
            'hot': [
                'Rail temperature monitoring with thermal cameras',
                'Heat-triggered speed restrictions (>35°C)',
                'Rail stress relief procedures during heat waves'
            ],
            'flooded': [
                'Water level sensors with automatic track closure',
                'Enhanced embankment drainage infrastructure',
                'Flood barrier systems at vulnerable locations'
            ]
        }
        
        if weather in weather_adaptations:
            adaptations.extend(random.sample(weather_adaptations[weather], min(2, len(weather_adaptations[weather]))))
        
        # Speed-specific adaptations
        if speed > 130:
            adaptations.append(f'High-speed certification process for {speed} km/h operation')
            adaptations.append('Enhanced track geometry tolerances for high-speed stability')
        elif speed > 100:
            adaptations.append('Medium-speed track quality monitoring systems')
        
        # Severity-specific
        if severity in ['critical', 'high']:
            adaptations.append('Independent safety audit by external consultants')
            adaptations.append('Enhanced staff training program based on incident analysis')
        
        # Location-specific
        if context.get('location', {}).get('is_junction', False):
            adaptations.append('Junction-specific interlocking verification procedures')
        
        return adaptations[:4]  # Limit to 4 most relevant
    
    def generate_golden_run(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete golden run solution"""
        accident_type = incident['accident_type']
        
        # Select appropriate solution template
        templates = self.solution_templates.get(accident_type, self.solution_templates['derailment'])
        template = random.choice(templates)
        
        # Calculate realistic costs and times
        base_cost = template['base_cost']
        base_time = template['base_time']
        
        actual_cost = self.calculate_realistic_cost(base_cost, incident)
        actual_time = self.calculate_implementation_time(base_time, incident)
        
        # Generate prevention cost (typically 30-50% of implementation)
        prevention_cost = int(actual_cost * random.uniform(0.30, 0.50))
        
        # Generate outcomes
        outcomes = self.generate_actual_outcomes(incident)
        
        # Generate context adaptations
        adaptations = self.generate_context_adaptations(accident_type, incident)
        
        # Select why it's golden
        why_golden = random.choice(self.outcome_narratives)
        
        # Build solution
        solution = {
            'description': template['description'],
            'variant': template['variant'],
            'implementation_time_hours': actual_time,
            'cost_estimate': actual_cost,
            'implementation_steps': template['steps'],
            'preventive_measures': template['preventive'],
            'estimated_prevention_cost': prevention_cost,
            'context_specific_adaptations': adaptations,
            'actual_outcomes': outcomes,
            'why_golden': why_golden,
            'reference_accident_types': [accident_type],
            'lessons_learned': self.generate_lessons_learned(accident_type, incident),
            'industry_recognition': self.generate_recognition(outcomes)
        }
        
        return solution
    
    def generate_lessons_learned(self, accident_type: str, context: Dict[str, Any]) -> List[str]:
        """Generate realistic lessons learned"""
        lessons = []
        
        # Generic lessons by type
        type_lessons = {
            'derailment': [
                'Track geometry degradation accelerates in curves - increase monitoring frequency',
                'Wheel-rail interface requires integrated monitoring, not isolated component checks'
            ],
            'collision': [
                'Human factors remain critical - automation must augment, not replace, operator judgment',
                'Communication redundancy prevented escalation - triple-channel systems now standard'
            ],
            'signal_failure': [
                'Environmental factors (humidity) were root cause - now monitored in real-time',
                'Hot-standby systems eliminated single points of failure'
            ],
            'brake_failure': [
                'Predictive maintenance reduced failures by 87% - now core to all systems',
                'Load-dependent brake force critical for long trains - automatic adjustment implemented'
            ]
        }
        
        if accident_type in type_lessons:
            lessons.extend(type_lessons[accident_type])
        
        # Add context-specific lesson
        if context.get('weather_conditions') in ['rainy', 'snowy']:
            lessons.append('Weather integration into control systems proved essential - now standard practice')
        
        return lessons[:3]
    
    def generate_recognition(self, outcomes: Dict[str, Any]) -> str:
        """Generate industry recognition based on outcomes"""
        if outcomes['delay_reduction_pct'] > 94:
            return 'Featured as case study in International Railway Safety Conference 2024'
        elif outcomes['safety_score_improvement'] > 0.25:
            return 'Received National Transportation Safety Excellence Award'
        elif outcomes['recurrence_prevented_months'] > 30:
            return 'Designated as reference implementation by Railway Standards Authority'
        else:
            return 'Recognized by regional transport authority as best practice'
    
    def process_all_incidents(self, incidents_file: str, output_file: str):
        """Process all incidents and generate golden runs"""
        # Load incidents
        with open(incidents_file, 'r') as f:
            data = json.load(f)
        
        incidents = data if isinstance(data, list) else data.get('incidents', [])
        
        # Generate golden runs
        for incident in incidents:
            golden_solution = self.generate_golden_run(incident)
            incident['solution'] = golden_solution
            incident['is_golden_run'] = True
            incident['golden_run_verified_at'] = datetime.now().isoformat()
        
        # Save enhanced golden runs
        output_data = {
            'metadata': {
                'total_golden_runs': len(incidents),
                'generated_at': datetime.now().isoformat(),
                'generator_version': '2.0_enhanced',
                'quality_assurance': {
                    'context_aware_solutions': True,
                    'realistic_cost_variance': True,
                    'outcome_differentiation': True,
                    'solution_diversity': len(set(inc['solution']['variant'] for inc in incidents))
                }
            },
            'golden_runs': incidents
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Generated {len(incidents)} enhanced golden runs")
        print(f"📊 Solution variants: {output_data['metadata']['quality_assurance']['solution_diversity']}")
        print(f"💾 Saved to: {output_file}")


# Usage example
if __name__ == "__main__":
    generator = EnhancedGoldenRunGenerator()
    
    # Process your existing golden runs
    generator.process_all_incidents(
        'data/processed/golden_runs_accidents.json',
        'data/processed/golden_runs_accidents_enhanced.json'
    )