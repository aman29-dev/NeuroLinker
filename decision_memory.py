"""
Human-AI Memory Continuity System
Core decision memory model and storage management
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib


class MemoryLayer(Enum):
    """Privacy levels for decision memory"""
    PRIVATE = "private"  # Visible only to user
    SHAREABLE = "shareable"  # Can be shared with AI for context
    PUBLIC = "public"  # Can be exported/shared


@dataclass
class Constraint:
    """Represents a constraint on a decision"""
    category: str  # time, cost, risk, emotional, etc.
    description: str
    severity: str  # low, medium, high
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Alternative:
    """Represents an alternative option considered"""
    option: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    rejected_reason: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Decision:
    """Core decision record with full context"""
    id: str
    title: str
    description: str
    goal: str  # User's intent/goal
    constraints: List[Constraint]
    alternatives: List[Alternative]
    final_choice: str
    reasoning: str  # Why this choice was made
    expected_outcome: Optional[str] = None
    related_decisions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    memory_layer: MemoryLayer = MemoryLayer.PRIVATE
    tags: List[str] = field(default_factory=list)
    reflection: Optional[str] = None  # User's reflection on the decision
    outcome_status: Optional[str] = None  # pending, completed, reviewing
    
    def to_dict(self):
        data = asdict(self)
        data['constraints'] = [c.to_dict() if isinstance(c, Constraint) else c for c in self.constraints]
        data['alternatives'] = [a.to_dict() if isinstance(a, Alternative) else a for a in self.alternatives]
        data['memory_layer'] = self.memory_layer.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Reconstruct Decision from dictionary"""
        data = data.copy()
        
        # Convert constraints
        if data.get('constraints'):
            data['constraints'] = [
                Constraint(**c) if isinstance(c, dict) else c 
                for c in data['constraints']
            ]
        
        # Convert alternatives
        if data.get('alternatives'):
            data['alternatives'] = [
                Alternative(**a) if isinstance(a, dict) else a 
                for a in data['alternatives']
            ]
        
        # Convert memory_layer
        if isinstance(data.get('memory_layer'), str):
            data['memory_layer'] = MemoryLayer(data['memory_layer'])
        
        return cls(**data)


class DecisionMemoryStore:
    """Persistent storage and management of decision memories"""
    
    def __init__(self, file_path: str = "human_ai_memory.json"):
        self.file_path = file_path
        self.decisions: Dict[str, Decision] = {}
        self.load_from_file()
    
    def load_from_file(self):
        """Load decisions from JSON file"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Handle old format
                        for item in data:
                            decision = Decision.from_dict(item)
                            self.decisions[decision.id] = decision
                    elif isinstance(data, dict):
                        for decision_id, decision_data in data.items():
                            decision = Decision.from_dict(decision_data)
                            self.decisions[decision_id] = decision
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading decisions: {e}")
                self.decisions = {}
    
    def save_to_file(self):
        """Persist decisions to JSON file"""
        data = {
            decision_id: decision.to_dict() 
            for decision_id, decision in self.decisions.items()
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_id(self, title: str) -> str:
        """Generate unique decision ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_obj = hashlib.md5((title + timestamp).encode())
        return f"dec_{hash_obj.hexdigest()[:8]}"
    
    def add_decision(self, decision: Decision) -> str:
        """Add a new decision to memory"""
        if not decision.id:
            decision.id = self.generate_id(decision.title)
        
        self.decisions[decision.id] = decision
        self.save_to_file()
        return decision.id
    
    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """Retrieve a specific decision"""
        return self.decisions.get(decision_id)
    
    def update_decision(self, decision_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing decision"""
        if decision_id not in self.decisions:
            return False
        
        decision = self.decisions[decision_id]
        decision.updated_at = datetime.now().isoformat()
        
        for key, value in updates.items():
            if hasattr(decision, key):
                setattr(decision, key, value)
        
        self.save_to_file()
        return True
    
    def delete_decision(self, decision_id: str) -> bool:
        """Delete a decision (with user confirmation in UI)"""
        if decision_id in self.decisions:
            del self.decisions[decision_id]
            self.save_to_file()
            return True
        return False
    
    def get_all_decisions(self, memory_layer: Optional[MemoryLayer] = None) -> List[Decision]:
        """Get all decisions, optionally filtered by memory layer"""
        decisions = list(self.decisions.values())
        if memory_layer:
            decisions = [d for d in decisions if d.memory_layer == memory_layer]
        return sorted(decisions, key=lambda d: d.created_at, reverse=True)
    
    def search_decisions(self, query: str) -> List[Decision]:
        """Search decisions by title, description, tags"""
        query_lower = query.lower()
        results = []
        
        for decision in self.decisions.values():
            if (query_lower in decision.title.lower() or
                query_lower in decision.description.lower() or
                any(query_lower in tag.lower() for tag in decision.tags)):
                results.append(decision)
        
        return sorted(results, key=lambda d: d.created_at, reverse=True)
    
    def get_related_decisions(self, decision_id: str) -> List[Decision]:
        """Get decisions related to a specific decision"""
        decision = self.get_decision(decision_id)
        if not decision:
            return []
        
        related = []
        for rel_id in decision.related_decisions:
            rel_decision = self.get_decision(rel_id)
            if rel_decision:
                related.append(rel_decision)
        
        return related
    
    def link_decisions(self, decision_id1: str, decision_id2: str):
        """Link two related decisions"""
        if decision_id1 in self.decisions and decision_id2 in self.decisions:
            if decision_id2 not in self.decisions[decision_id1].related_decisions:
                self.decisions[decision_id1].related_decisions.append(decision_id2)
            if decision_id1 not in self.decisions[decision_id2].related_decisions:
                self.decisions[decision_id2].related_decisions.append(decision_id1)
            self.save_to_file()
    
    def get_decision_categories(self) -> Dict[str, int]:
        """Get count of decisions by tag/category"""
        categories = {}
        for decision in self.decisions.values():
            for tag in decision.tags:
                categories[tag] = categories.get(tag, 0) + 1
        return categories
    
    def get_constraint_patterns(self) -> Dict[str, int]:
        """Analyze constraint patterns across decisions"""
        patterns = {}
        for decision in self.decisions.values():
            for constraint in decision.constraints:
                key = f"{constraint.category} ({constraint.severity})"
                patterns[key] = patterns.get(key, 0) + 1
        return patterns


class AIReasoningEngine:
    """Provides context-aware suggestions based on decision history"""
    
    def __init__(self, memory_store: DecisionMemoryStore):
        self.store = memory_store
    
    def analyze_constraint_patterns(self) -> Dict[str, Any]:
        """Identify recurring constraints in user decisions"""
        return self.store.get_constraint_patterns()
    
    def find_similar_decisions(self, current_goal: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find decisions with similar goals or context - only returns truly similar decisions"""
        similar = []
        current_goal_lower = current_goal.lower()
        
        # Early exit if no decisions
        if not self.store.get_all_decisions():
            return []
        
        for decision in self.store.get_all_decisions():
            # Score decisions by relevance
            relevance_score = 0
            
            # Full goal match (highest priority)
            if current_goal_lower in decision.goal.lower() or decision.goal.lower() in current_goal_lower:
                relevance_score += 100
            
            # Word overlap in goal (words > 3 chars)
            current_words = set(w.lower() for w in current_goal_lower.split() if len(w) > 3)
            decision_words = set(w.lower() for w in decision.goal.lower().split() if len(w) > 3)
            word_overlap = len(current_words & decision_words)
            relevance_score += word_overlap * 15
            
            # Check if situation type matches via tags
            current_tags = set(w.lower() for w in current_goal_lower.split() if len(w) > 3)
            decision_tags = set(t.lower() for t in decision.tags)
            tag_overlap = len(current_tags & decision_tags)
            relevance_score += tag_overlap * 10
            
            # Check if description contains relevant keywords
            if current_goal_lower in decision.description.lower() or current_goal_lower in decision.reasoning.lower():
                relevance_score += 50
            
            # Only include if relevance score is meaningful (not just random match)
            # Minimum threshold: 10 points (prevents every decision showing up)
            if relevance_score >= 10:
                similar.append({
                    'decision': decision,
                    'relevance': relevance_score,
                    'relevance_reason': 'goal_match' if relevance_score > 50 else 'related_experience'
                })
        
        # Sort by relevance (highest first), then by recency
        similar.sort(key=lambda x: (-x['relevance'], x['decision'].created_at), reverse=True)
        return similar[:limit]
    
    def generate_contextual_suggestion(self, current_situation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI suggestion based on past decisions"""
        suggestions = {
            'learned_constraints': [],
            'pattern_insights': [],
            'past_reasoning': [],
            'ai_recommendation': '',
            'has_similar': False,
            'cautions': []
        }
        
        current_goal = current_situation.get('goal', '')
        
        # Find similar past decisions
        similar = self.find_similar_decisions(current_goal, limit=5)
        suggestions['has_similar'] = len(similar) > 0
        
        # Extract past decisions info
        for item in similar:
            decision = item['decision']
            suggestions['past_reasoning'].append({
                'title': decision.title,
                'goal': decision.goal,
                'final_choice': decision.final_choice,
                'reasoning': decision.reasoning,
                'constraints_faced': [
                    f"{c.category}: {c.description}" 
                    for c in decision.constraints
                ],
                'relevance': item['relevance_reason']
            })
        
        # Analyze constraint patterns across ALL decisions
        constraint_patterns = self.analyze_constraint_patterns()
        if constraint_patterns:
            top_constraints = sorted(constraint_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
            suggestions['learned_constraints'] = [
                f"📌 {constraint} (faced {count}x)"
                for constraint, count in top_constraints
            ]
        
        # Generate AI recommendation based on findings
        if suggestions['has_similar'] and suggestions['past_reasoning']:
            # Found similar decisions - provide personalized advice
            past_choices = [d['final_choice'] for d in suggestions['past_reasoning']]
            past_goals = [d['goal'] for d in suggestions['past_reasoning']]
            
            # Extract common themes
            themes = []
            for decision in [item['decision'] for item in similar]:
                for tag in decision.tags:
                    themes.append(tag)
            
            theme_str = ", ".join(set(themes)) if themes else "your decision patterns"
            
            # Get most common constraint safely
            most_common_constraint = list(constraint_patterns.keys())[0] if constraint_patterns else "multiple factors"
            
            suggestions['ai_recommendation'] = f"""
✅ **Found Similar Past Decisions!**

Based on your {len(self.store.get_all_decisions())} past decisions, your situation is **similar** to decisions you've made before.

**Your Pattern:** You've made decisions about {theme_str}. 
Your approach typically focuses on {most_common_constraint.lower()}.

**Key Learnings:**
1. **Previous approach:** {past_choices[0] if past_choices else 'Multiple solutions considered'}
2. **Key constraint:** {most_common_constraint}
3. **Your reasoning:** {past_goals[0] if past_goals else 'Consider all factors'}

**Recommendation:** Review the similar decisions below. Notice:
- How you've handled similar constraints before
- What trade-offs you made
- The reasoning that led to your choice
- Apply this framework to your current situation
"""
        else:
            # No similar decisions found - provide general guidance
            all_decisions = self.store.get_all_decisions()
            
            if len(all_decisions) > 0:
                suggestions['ai_recommendation'] = f"""
ℹ️ **No Directly Similar Past Decisions**

Your current situation doesn't closely match your past decisions, but you can still learn from them!

**What I can tell you:**
- You've made {len(all_decisions)} decision(s) before
- You often face: {constraint_patterns.get(list(constraint_patterns.keys())[0], 'multiple constraints') if constraint_patterns else 'various constraints'}

**My Suggestions:**
1. **Review your decision framework** - Look at any past decision to see how you've approached constraints
2. **Think about your patterns** - What factors do you usually prioritize?
3. **Document this new decision** - Add it to your memory so future decisions can learn from it
4. **Trust your judgment** - You've made good decisions before; apply that same thinking here

This new decision will help me provide better suggestions next time!
"""
            else:
                suggestions['ai_recommendation'] = """
📝 **No Decision History Yet**

You haven't recorded any decisions yet. To get AI suggestions, start by:

1. **Record your current situation** as a new decision
2. **Document the details:**
   - Your goal and what you're trying to achieve
   - Constraints you face (time, cost, resources, etc.)
   - Alternatives you're considering
   - Why you choose one option over others

3. **Record similar decisions in the future** - The more decisions you log, the better my suggestions become!

**Start by recording your first decision →** Go back and use "Record Decision" option.
"""
        
        # Generate pattern insights
        if len(self.store.get_all_decisions()) > 2:
            suggestions['pattern_insights'] = [
                "✓ You have enough decision history. Patterns are emerging.",
                f"✓ You tend to face similar constraints - be proactive about them",
                f"✓ Link related decisions to understand your evolution"
            ]
        elif len(self.store.get_all_decisions()) > 0:
            suggestions['pattern_insights'] = [
                f"📝 You have {len(self.store.get_all_decisions())} decision recorded",
                "📝 Record more decisions to unlock pattern analysis",
                "📝 Document constraints & alternatives for better insights"
            ]
        else:
            suggestions['pattern_insights'] = [
                "🆕 Start recording decisions to unlock AI insights"
            ]
        
        return suggestions
    
    def explain_past_decision(self, decision_id: str) -> Dict[str, Any]:
        """Generate explanation of why a past decision was made"""
        decision = self.store.get_decision(decision_id)
        if not decision:
            return {}
        
        explanation = {
            'goal': decision.goal,
            'constraints_faced': [
                {
                    'category': c.category,
                    'description': c.description,
                    'severity': c.severity
                }
                for c in decision.constraints
            ],
            'alternatives_considered': [
                {
                    'option': a.option,
                    'why_rejected': a.rejected_reason or 'Not selected'
                }
                for a in decision.alternatives
            ],
            'final_reasoning': decision.reasoning,
            'expected_outcome': decision.expected_outcome
        }
        
        return explanation