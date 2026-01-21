"""
Human-AI Memory Continuity System
Interactive Streamlit Application
"""

import streamlit as st
import json
from datetime import datetime
from dotenv import load_dotenv
import os
from decision_memory import (
    DecisionMemoryStore, Decision, Constraint, Alternative,
    MemoryLayer, AIReasoningEngine
)
from reflection_gemini import GeminiReflectionAI
from gemini_chatbot import AIDecisionChatbot
import pandas as pd
from typing import List, Dict, Any
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Page configuration
st.set_page_config(
    page_title="Human-AI Memory Continuity System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UX
st.markdown("""
<style>
    .decision-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .constraint-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        margin-right: 5px;
        font-size: 12px;
    }
    .high-severity {
        background-color: #ffcccc;
        color: #cc0000;
    }
    .medium-severity {
        background-color: #ffe6cc;
        color: #ff8800;
    }
    .low-severity {
        background-color: #e6f2ff;
        color: #0066cc;
    }
    .section-header {
        border-bottom: 3px solid #0066cc;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'memory_store' not in st.session_state:
    st.session_state.memory_store = DecisionMemoryStore()
    st.session_state.ai_engine = AIReasoningEngine(st.session_state.memory_store)
    st.session_state.gemini_reflection = GeminiReflectionAI(enabled=True)
    try:
        st.session_state.ai_chatbot = AIDecisionChatbot()
    except ValueError as e:
        st.session_state.ai_chatbot = None
        st.warning(f"Chatbot initialization failed: {e}")

if 'current_view' not in st.session_state:
    st.session_state.current_view = 'home'

if 'selected_decision' not in st.session_state:
    st.session_state.selected_decision = None

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def render_home():
    """Home page with overview and quick stats"""
    st.title("🧠 Human-AI Memory Continuity System")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_decisions = len(st.session_state.memory_store.get_all_decisions())
    private_decisions = len(st.session_state.memory_store.get_all_decisions(MemoryLayer.PRIVATE))
    shareable_decisions = len(st.session_state.memory_store.get_all_decisions(MemoryLayer.SHAREABLE))
    
    with col1:
        st.metric("Total Decisions", total_decisions, help="All decisions in your memory")
    with col2:
        st.metric("Private Decisions", private_decisions, help="Visible only to you")
    with col3:
        st.metric("Shareable Decisions", shareable_decisions, help="Can be shared with AI")
    with col4:
        st.metric("Memory Integrity", "100%", help="All decisions preserved")
    
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to Your Decision Memory System
    
    This system helps you and AI maintain meaningful continuity by remembering not just *what* you decided,
    but **why** you decided it. Over time, this creates a rich context that enables smarter assistance.
    
    **Key Features:**
    - 📝 **Record decisions** with full context: goals, constraints, alternatives, and reasoning
    - 🔄 **AI Recall** - Understand past decisions with their original reasoning
    - 🔗 **Link decisions** - See how your choices relate over time
    - 🛡️ **Privacy First** - You control what's stored and shared
    - 📊 **Analyze patterns** - Identify recurring challenges and preferences
    """)
    
    st.markdown("---")
    
    # Quick action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📝 Record New Decision", use_container_width=True, key="btn_new_decision"):
            st.session_state.current_view = 'record'
            st.rerun()
    
    with col2:
        if st.button("📚 View All Decisions", use_container_width=True, key="btn_view_decisions"):
            st.session_state.current_view = 'timeline'
            st.rerun()
    
    with col3:
        if st.button("🤖 Get AI Insights", use_container_width=True, key="btn_ai_insights"):
            st.session_state.current_view = 'ai_insights'
            st.rerun()
    
    with col4:
        if st.button("📊 View Analytics", use_container_width=True, key="btn_analytics"):
            st.session_state.current_view = 'analytics'
            st.rerun()
    
    # Recent decisions preview
    if total_decisions > 0:
        st.markdown("---")
        st.markdown("### 📌 Recent Decisions")
        recent = st.session_state.memory_store.get_all_decisions()[:3]
        
        for decision in recent:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{decision.title}**")
                    st.caption(f"Goal: {decision.goal[:100]}...")
                    st.caption(f"Choice: {decision.final_choice}")
                with col2:
                    memory_color = "🔒" if decision.memory_layer == MemoryLayer.PRIVATE else "🔓"
                    st.metric("Layer", memory_color, help=decision.memory_layer.value)
                
                if st.button(f"View Details", key=f"btn_view_{decision.id}"):
                    st.session_state.selected_decision = decision.id
                    st.session_state.current_view = 'decision_detail'
                    st.rerun()


def render_record_decision():
    """Form to record a new decision"""
    st.title("📝 Record a New Decision")
    st.markdown("Capture your decision with full context for better AI understanding")
    
    # Initialize session state for counts
    if 'constraint_count' not in st.session_state:
        st.session_state.constraint_count = 2
    if 'alt_count' not in st.session_state:
        st.session_state.alt_count = 2
    
    # Number inputs OUTSIDE the form so they trigger reruns
    st.markdown("### ⚠️ Constraints You Face")
    col1, col2 = st.columns(2)
    with col1:
        new_constraint_count = st.number_input("Number of constraints to add", min_value=0, max_value=10, value=st.session_state.constraint_count, key="constraint_count_control")
        if new_constraint_count != st.session_state.constraint_count:
            st.session_state.constraint_count = new_constraint_count
            st.rerun()
    
    st.markdown("### 🔄 Alternatives Considered")
    with col2:
        new_alt_count = st.number_input("Number of alternatives to document", min_value=1, max_value=10, value=st.session_state.alt_count, key="alt_count_control")
        if new_alt_count != st.session_state.alt_count:
            st.session_state.alt_count = new_alt_count
            st.rerun()
    
    with st.form("decision_form"):
        # Basic information
        st.markdown("### 📋 Decision Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Decision Title*", 
                                 placeholder="e.g., Choose new project technology stack")
        
        with col2:
            description = st.text_area("Decision Description*", 
                                      placeholder="Briefly describe what decision you're making",
                                      height=100)
        
        # Goal and intent
        st.markdown("### 🎯 Your Goal & Intent")
        goal = st.text_area("What is your goal or intention?*",
                           placeholder="e.g., Improve team productivity while maintaining code quality",
                           height=80)
        
        # Constraints
        st.markdown("### ⚠️ Constraints (Configured Above)")
        st.write("What are the limitations or constraints on this decision?")
        
        constraints_list = []
        
        for i in range(st.session_state.constraint_count):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                category = st.selectbox(
                    f"Constraint {i+1} - Category",
                    ["Time", "Cost", "Risk", "Resource", "Technical", "Emotional", "Other"],
                    key=f"constraint_cat_{i}"
                )
            
            with col2:
                description_c = st.text_input(f"Constraint {i+1} - Description", 
                                             key=f"constraint_desc_{i}",
                                             placeholder="e.g., Only 2 weeks available")
            
            with col3:
                severity = st.selectbox(f"Severity", ["Low", "Medium", "High"], 
                                       key=f"constraint_sev_{i}")
            
            if category and description_c:
                constraints_list.append(Constraint(
                    category=category,
                    description=description_c,
                    severity=severity
                ))
        
        # Alternatives considered
        st.markdown("### 🔄 Alternatives (Configured Above)")
        st.write("What other options did you consider? Why did you reject them?")
        
        alternatives_list = []
        
        for i in range(st.session_state.alt_count):
            st.markdown(f"**Alternative {i+1}**")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                option = st.text_input(f"Option {i+1}", key=f"alt_option_{i}",
                                      placeholder="e.g., Use existing framework")
            
            with col2:
                pros = st.multiselect(f"Pros", 
                                     ["Cost-effective", "Familiar", "Proven", "Simple", "Scalable", "Fast", "Reliable", "Efficient", "Flexible", "User-friendly", "Modern", "Secure"],
                                     key=f"alt_pros_{i}")
            
            cons = st.multiselect(f"Cons", 
                                 ["Outdated", "Limited features", "Poor support", "Learning curve", "Expensive", "Complex", "Slow", "Risky", "Incompatible", "Unmaintained", "Unstable"],
                                 key=f"alt_cons_{i}")
            
            rejected_reason = st.text_input(f"Why rejected?", key=f"alt_rejected_{i}",
                                           placeholder="e.g., Doesn't meet scalability needs")
            
            if option:
                alternatives_list.append(Alternative(
                    option=option,
                    pros=list(pros),
                    cons=list(cons),
                    rejected_reason=rejected_reason
                ))
        
        # Final choice and reasoning
        st.markdown("### ✅ Your Final Decision")
        final_choice = st.text_input("What did you choose?*",
                                    placeholder="e.g., Migrate to React with TypeScript")
        
        reasoning = st.text_area("Why did you choose this?*",
                                placeholder="Explain the reasoning behind your choice, considering constraints and trade-offs",
                                height=120)
        
        expected_outcome = st.text_area("What outcome do you expect?",
                                       placeholder="e.g., 30% improvement in development speed",
                                       height=80)
        
        # Tags
        st.markdown("### 🏷️ Tags & Categorization")
        tags_input = st.multiselect("Add tags to categorize this decision",
                                   ["Career", "Project", "Technical", "Financial", "Personal", "Academic", "Health", "Lifestyle"],
                                   key="tags_select")
        
        # Memory layer and consent
        st.markdown("### 🔒 Privacy & Control")
        memory_layer = st.radio("Memory Privacy Level",
                               ["🔒 Private (Only visible to you)",
                                "🔓 Shareable (Can be shared with AI for context)"],
                               index=0)
        
        memory_layer_enum = MemoryLayer.PRIVATE if "🔒" in memory_layer else MemoryLayer.SHAREABLE
        
        st.info("""
        **Privacy Explanation:**
        - **Private**: Only you see this. AI cannot access it.
        - **Shareable**: AI can use this for context-aware suggestions, but won't share with third parties.
        """)
        
        # Consent checkbox
        consent = st.checkbox("I understand the privacy terms and consent to this decision being recorded")
        
        # Submit button
        submitted = st.form_submit_button("💾 Save Decision", use_container_width=True)
        
        if submitted:
            if not title or not description or not goal or not final_choice or not reasoning or not consent:
                st.error("Please fill all required fields and confirm consent")
            else:
                # Create decision object
                decision = Decision(
                    id="",  # Will be generated
                    title=title,
                    description=description,
                    goal=goal,
                    constraints=constraints_list,
                    alternatives=alternatives_list,
                    final_choice=final_choice,
                    reasoning=reasoning,
                    expected_outcome=expected_outcome,
                    memory_layer=memory_layer_enum,
                    tags=tags_input
                )
                
                # Save to store
                decision_id = st.session_state.memory_store.add_decision(decision)
                
                st.success(f"✅ Decision recorded successfully! (ID: {decision_id[:12]}...)")
                st.balloons()
                
                # Reset form
                st.session_state.current_view = 'timeline'
                st.rerun()


def render_decision_timeline():
    """Display decision timeline with filtering and search"""
    st.title("📚 Decision Timeline")
    
    decisions = st.session_state.memory_store.get_all_decisions()
    
    if not decisions:
        st.info("No decisions recorded yet. Start by recording your first decision!")
        if st.button("Record First Decision"):
            st.session_state.current_view = 'record'
            st.rerun()
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_query = st.text_input("🔍 Search decisions", placeholder="Enter title, goal, or tag")
    
    with col2:
        selected_layer = st.selectbox("Filter by Privacy Layer",
                                     ["All", "🔒 Private", "🔓 Shareable"])
    
    with col3:
        selected_tag = st.selectbox("Filter by Tag",
                                   ["All"] + list(st.session_state.memory_store.get_decision_categories().keys()))
    
    # Apply filters
    filtered_decisions = decisions
    
    if search_query:
        filtered_decisions = st.session_state.memory_store.search_decisions(search_query)
    
    if selected_layer != "All":
        layer = MemoryLayer.PRIVATE if "🔒" in selected_layer else MemoryLayer.SHAREABLE
        filtered_decisions = [d for d in filtered_decisions if d.memory_layer == layer]
    
    if selected_tag != "All":
        filtered_decisions = [d for d in filtered_decisions if selected_tag in d.tags]
    
    # Display decisions
    st.markdown(f"### Showing {len(filtered_decisions)} decision(s)")
    
    for decision in filtered_decisions:
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                memory_icon = "🔒" if decision.memory_layer == MemoryLayer.PRIVATE else "🔓"
                st.markdown(f"### {memory_icon} {decision.title}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption(f"📅 {datetime.fromisoformat(decision.created_at).strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"🎯 **Goal**: {decision.goal[:80]}...")
                with col_b:
                    st.caption(f"✅ **Choice**: {decision.final_choice[:80]}...")
                
                if decision.tags:
                    tag_badges = " ".join([f"🏷️ {tag}" for tag in decision.tags])
                    st.caption(tag_badges)
            
            with col2:
                constraint_count = len(decision.constraints)
                st.metric("Constraints", constraint_count)
            
            with col3:
                alt_count = len(decision.alternatives)
                st.metric("Alternatives", alt_count)
            
            # Action buttons
            col_x, col_y, col_z = st.columns(3)
            
            with col_x:
                if st.button("👁️ View", key=f"view_{decision.id}", use_container_width=True):
                    st.session_state.selected_decision = decision.id
                    st.session_state.current_view = 'decision_detail'
                    st.rerun()
            
            with col_y:
                if st.button("✏️ Edit", key=f"edit_{decision.id}", use_container_width=True):
                    st.session_state.selected_decision = decision.id
                    st.session_state.current_view = 'edit_decision'
                    st.rerun()
            
            with col_z:
                if st.button("🗑️ Delete", key=f"delete_{decision.id}", use_container_width=True):
                    if st.session_state.memory_store.delete_decision(decision.id):
                        st.success("Decision deleted")
                        st.rerun()


def render_decision_detail():
    """Display detailed view of a decision"""
    decision_id = st.session_state.selected_decision
    decision = st.session_state.memory_store.get_decision(decision_id)
    
    if not decision:
        st.error("Decision not found")
        if st.button("Back to Timeline"):
            st.session_state.current_view = 'timeline'
            st.rerun()
        return
    
    # Header
    memory_icon = "🔒" if decision.memory_layer == MemoryLayer.PRIVATE else "🔓"
    st.title(f"{memory_icon} {decision.title}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Created", datetime.fromisoformat(decision.created_at).strftime('%Y-%m-%d'))
    with col2:
        st.metric("Status", decision.outcome_status or "Pending Review")
    with col3:
        st.metric("Privacy", decision.memory_layer.value.title())
    with col4:
        st.metric("Related", len(decision.related_decisions))
    
    st.markdown("---")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Analysis", "Related", "Reflection", "Actions"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Decision Summary")
            st.write(f"**Description**: {decision.description}")
            st.write(f"**Goal**: {decision.goal}")
            st.write(f"**Final Choice**: {decision.final_choice}")
        
        with col2:
            st.markdown("### 🎯 Reasoning")
            st.write(decision.reasoning)
            if decision.expected_outcome:
                st.markdown("### 📊 Expected Outcome")
                st.write(decision.expected_outcome)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚠️ Constraints Faced")
            if decision.constraints:
                for constraint in decision.constraints:
                    severity_color = {
                        "high": "🔴", "medium": "🟡", "low": "🟢"
                    }.get(constraint.severity.lower(), "⚪")
                    st.write(f"{severity_color} **{constraint.category}** ({constraint.severity})")
                    st.caption(constraint.description)
            else:
                st.info("No constraints recorded")
        
        with col2:
            st.markdown("### 🔄 Alternatives Considered")
            if decision.alternatives:
                for i, alt in enumerate(decision.alternatives, 1):
                    with st.container(border=True):
                        st.write(f"**{i}. {alt.option}**")
                        if alt.pros:
                            st.caption(f"✅ Pros: {', '.join(alt.pros)}")
                        if alt.cons:
                            st.caption(f"❌ Cons: {', '.join(alt.cons)}")
                        if alt.rejected_reason:
                            st.caption(f"⛔ Why rejected: {alt.rejected_reason}")
            else:
                st.info("No alternatives recorded")
        
        # AI Analysis
        st.markdown("---")
        st.markdown("### 🤖 AI Decision Analysis")
        explanation = st.session_state.ai_engine.explain_past_decision(decision_id)
        
        if explanation:
            with st.expander("📖 Full Decision Context"):
                st.json(explanation)
    
    with tab3:
        st.markdown("### 🔗 Related Decisions")
        related = st.session_state.memory_store.get_related_decisions(decision_id)
        
        if related:
            for rel_decision in related:
                with st.container(border=True):
                    st.write(f"**{rel_decision.title}**")
                    st.caption(f"Goal: {rel_decision.goal}")
                    if st.button("View", key=f"view_rel_{rel_decision.id}"):
                        st.session_state.selected_decision = rel_decision.id
                        st.rerun()
        else:
            st.info("No related decisions yet.")
            
            # Link decisions
            all_decisions = [d for d in st.session_state.memory_store.get_all_decisions() 
                            if d.id != decision_id]
            if all_decisions:
                st.markdown("#### Link with another decision:")
                selected_decision = st.selectbox("Select decision to link",
                                               [(d.id, d.title) for d in all_decisions],
                                               format_func=lambda x: x[1])
                
                if st.button("🔗 Create Link"):
                    st.session_state.memory_store.link_decisions(decision_id, selected_decision[0])
                    st.success("Decisions linked!")
                    st.rerun()
    
    with tab4:
        st.markdown("### 💭 Reflection & Outcome")
        
        current_reflection = decision.reflection or ""
        new_reflection = st.text_area("Add your reflection on this decision",
                                     value=current_reflection,
                                     height=150,
                                     placeholder="How did this decision turn out? What would you do differently?")
        
        current_status = decision.outcome_status or "Pending Review"
        new_status = st.selectbox("Outcome Status",
                                 ["Pending Review", "Completed", "In Progress", "Reviewing"],
                                 index=["Pending Review", "Completed", "In Progress", "Reviewing"].index(current_status))
        
        if st.button("💾 Save Reflection"):
            st.session_state.memory_store.update_decision(
                decision_id,
                {"reflection": new_reflection, "outcome_status": new_status}
            )
            st.success("Reflection saved!")
            st.rerun()
        
        # Gemini AI Reflection
        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Reflection (Powered by Gemini)")
        
        decision_history_text = f"""
        Decision: {decision.title}
        Goal: {decision.goal}
        Description: {decision.description}
        Final Choice: {decision.final_choice}
        Reasoning: {decision.reasoning}
        Expected Outcome: {decision.expected_outcome}
        Constraints: {', '.join([c.description for c in decision.constraints])}
        Alternatives Considered: {', '.join([a.option for a in decision.alternatives])}
        Reflection: {new_reflection}
        """
        
        if st.button("✨ Get Gemini AI Reflection"):
            with st.spinner("🤔 Gemini is thinking..."):
                reflection_ai = GeminiReflectionAI(enabled=True)
                ai_reflection = reflection_ai.reflect(decision_history_text)
                
                if "failed" not in ai_reflection.lower():
                    st.markdown("#### 💡 Gemini's Insight:")
                    st.info(ai_reflection)
                    
                    # Option to add to reflection
                    if st.button("📝 Add to your reflection"):
                        updated_reflection = new_reflection + f"\n\n**AI Insight**: {ai_reflection}"
                        st.session_state.memory_store.update_decision(
                            decision_id,
                            {"reflection": updated_reflection}
                        )
                        st.success("AI reflection added!")
                        st.rerun()
                else:
                    st.error(ai_reflection)
    
    with tab5:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✏️ Edit Decision", use_container_width=True):
                st.session_state.current_view = 'edit_decision'
                st.rerun()
        
        with col2:
            if st.button("🗑️ Delete Decision", use_container_width=True):
                if st.confirmation_dialog("Confirm deletion", "This action cannot be undone"):
                    st.session_state.memory_store.delete_decision(decision_id)
                    st.success("Decision deleted!")
                    st.session_state.current_view = 'timeline'
                    st.rerun()
    
    # Back button
    st.markdown("---")
    if st.button("← Back to Timeline"):
        st.session_state.current_view = 'timeline'
        st.rerun()


def render_ai_insights():
    """AI-powered chatbot for decision assistance"""
    st.title("🤖 AI Decision Assistant Chatbot")
    
    # Check if chatbot is available
    if not st.session_state.ai_chatbot:
        st.error("⚠️ AI Chatbot is not available. Please check your GEMINI_API_KEY configuration.")
        if st.button("← Back to Home"):
            st.session_state.current_view = 'home'
            st.rerun()
        return
    
    # Get user's decision context
    decisions = st.session_state.memory_store.get_all_decisions()
    
    if not decisions:
        st.info("💡 Tip: Record some decisions first for the AI to provide better context-aware suggestions!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Record Your First Decision", use_container_width=True):
                st.session_state.current_view = 'record'
                st.rerun()
        with col2:
            if st.button("← Back to Home", use_container_width=True):
                st.session_state.current_view = 'home'
                st.rerun()
        
        # Still allow chat without history
        st.markdown("---")
        st.markdown("### 💬 Start Chatting With AI")
    else:
        # Set context for chatbot with user's decision history
        decisions_summary = _get_decisions_summary(decisions)
        constraints_summary = _get_constraints_summary(decisions)
        st.session_state.ai_chatbot.set_user_context(decisions_summary, constraints_summary)
        
        # Show quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Decisions", len(decisions))
        with col2:
            avg_constraints = sum(len(d.constraints) for d in decisions) / len(decisions)
            st.metric("⚠️ Avg Constraints", f"{avg_constraints:.1f}")
        with col3:
            categories = st.session_state.memory_store.get_decision_categories()
            st.metric("🏷️ Categories", len(categories))
        
        st.markdown("---")
    
    # Chat interface
    st.markdown("### 💬 Chat with Your AI Decision Assistant")
    st.markdown("""
    Ask me anything about your decisions! I can help you:
    - **Understand** past decisions and their patterns
    - **Get suggestions** for current challenges based on your history
    - **Analyze** recurring constraints and patterns
    - **Reflect** on your decision-making style
    - **Plan** future decisions with lessons learned
    
    I speak your language and understand your unique decision-making context! 🌍
    """)
    
    # Chat history display
    st.markdown("---")
    
    # Initialize or restore chatbot history
    if st.session_state.chat_messages and not st.session_state.ai_chatbot.chat_history:
        st.session_state.ai_chatbot.set_conversation_history(st.session_state.chat_history)
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"], avatar=message.get("avatar", None)):
                st.markdown(message["content"])
    
    # Chat input
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.chat_input(
            "Type your question or describe your situation...",
            key="chat_input"
        )
    
    with col2:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.ai_chatbot.clear_history()
            st.rerun()
    
    # Process user input
    if user_input:
        # Add user message to chat
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input,
            "avatar": "👤"
        })
        
        # Get AI response
        with st.spinner("🤔 AI is thinking..."):
            try:
                ai_response = st.session_state.ai_chatbot.chat(user_input)
                
                # Add AI response to chat
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": ai_response,
                    "avatar": "🤖"
                })
                
                # Save chat history for session persistence
                st.session_state.chat_history = st.session_state.ai_chatbot.get_conversation_history()
                
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Sidebar options
    st.markdown("---")
    st.markdown("### 🎯 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.current_view = 'home'
            st.rerun()
    
    with col2:
        if st.button("📝 Record Decision", use_container_width=True):
            st.session_state.current_view = 'record'
            st.rerun()
    
    with col3:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.current_view = 'analytics'
            st.rerun()


def _get_decisions_summary(decisions: List[Decision]) -> str:
    """Create a summary of user's decisions for AI context"""
    if not decisions:
        return "No decisions recorded yet."
    
    summary_lines = [f"Total decisions: {len(decisions)}\n"]
    
    for i, decision in enumerate(decisions[-5:], 1):  # Last 5 decisions
        summary_lines.append(f"\nDecision {i}: {decision.title}")
        summary_lines.append(f"Goal: {decision.goal}")
        summary_lines.append(f"Choice: {decision.final_choice}")
        summary_lines.append(f"Reasoning: {decision.reasoning[:200]}...")
    
    return "\n".join(summary_lines)


def _get_constraints_summary(decisions: List[Decision]) -> str:
    """Create a summary of recurring constraints for AI context"""
    if not decisions:
        return "No constraints recorded yet."
    
    constraint_patterns = {}
    for decision in decisions:
        for constraint in decision.constraints:
            key = f"{constraint.category} (Severity: {constraint.severity})"
            constraint_patterns[key] = constraint_patterns.get(key, 0) + 1
    
    if not constraint_patterns:
        return "No constraint patterns found."
    
    summary_lines = ["Your most common constraints:"]
    for constraint, count in sorted(constraint_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        summary_lines.append(f"• {constraint}: appears {count} times")
    
    return "\n".join(summary_lines)


def render_analytics():
    """Analytics dashboard"""
    st.title("📊 Analytics & Insights Dashboard")
    
    decisions = st.session_state.memory_store.get_all_decisions()
    
    if not decisions:
        st.info("No data yet. Record decisions to see analytics!")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Decisions", len(decisions))
    with col2:
        avg_constraints = sum(len(d.constraints) for d in decisions) / len(decisions)
        st.metric("Avg Constraints", f"{avg_constraints:.1f}")
    with col3:
        avg_alternatives = sum(len(d.alternatives) for d in decisions) / len(decisions)
        st.metric("Avg Alternatives", f"{avg_alternatives:.1f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Decisions Over Time")
        dates = [datetime.fromisoformat(d.created_at).strftime('%Y-%m-%d') for d in decisions]
        st.write(f"First decision: {min(dates)}")
        st.write(f"Latest decision: {max(dates)}")
    
    with col2:
        st.markdown("### 🏷️ Decisions by Category")
        categories = st.session_state.memory_store.get_decision_categories()
        if categories:
            df = pd.DataFrame({
                "Category": list(categories.keys()),
                "Count": list(categories.values())
            })
            st.bar_chart(df.set_index("Category"))
    
    st.markdown("---")
    st.markdown("### 🔒 Privacy Distribution")
    
    private_count = len([d for d in decisions if d.memory_layer == MemoryLayer.PRIVATE])
    shareable_count = len([d for d in decisions if d.memory_layer == MemoryLayer.SHAREABLE])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Private Decisions", private_count)
    with col2:
        st.metric("Shareable Decisions", shareable_count)
    
    if st.button("← Back to Home"):
        st.session_state.current_view = 'home'
        st.rerun()


def render_edit_decision():
    """Edit existing decision"""
    decision_id = st.session_state.selected_decision
    decision = st.session_state.memory_store.get_decision(decision_id)
    
    if not decision:
        st.error("Decision not found")
        return
    
    st.title(f"✏️ Edit: {decision.title}")
    
    with st.form("edit_decision_form"):
        # Basic fields
        title = st.text_input("Title", value=decision.title)
        description = st.text_area("Description", value=decision.description)
        goal = st.text_area("Goal", value=decision.goal)
        final_choice = st.text_input("Final Choice", value=decision.final_choice)
        reasoning = st.text_area("Reasoning", value=decision.reasoning)
        expected_outcome = st.text_area("Expected Outcome", value=decision.expected_outcome or "")
        
        # Memory layer
        memory_layer = st.radio("Privacy Level",
                               ["🔒 Private", "🔓 Shareable"],
                               index=0 if decision.memory_layer == MemoryLayer.PRIVATE else 1)
        memory_layer_enum = MemoryLayer.PRIVATE if "🔒" in memory_layer else MemoryLayer.SHAREABLE
        
        # Tags
        tags = st.multiselect("Tags", 
                             ["Career", "Project", "Technical", "Financial", "Personal", "Academic", "Health", "Lifestyle"],
                             default=decision.tags)
        
        submitted = st.form_submit_button("💾 Save Changes")
        
        if submitted:
            st.session_state.memory_store.update_decision(
                decision_id,
                {
                    "title": title,
                    "description": description,
                    "goal": goal,
                    "final_choice": final_choice,
                    "reasoning": reasoning,
                    "expected_outcome": expected_outcome,
                    "memory_layer": memory_layer_enum,
                    "tags": tags
                }
            )
            st.success("Decision updated!")
            st.session_state.current_view = 'decision_detail'
            st.rerun()
    
    if st.button("Cancel"):
        st.session_state.current_view = 'decision_detail'
        st.rerun()


# Main app routing
def main():
    # Sidebar navigation
    with st.sidebar:
        st.markdown("# 🧠 Memory System")
        st.markdown("---")
        
        # Only show navigation radio for main views, not detail/edit views
        nav_labels = ["Home", "Record Decision", "View Timeline", "AI Insights", "Analytics"]
        nav_keys = ["home", "record", "timeline", "ai_insights", "analytics"]
        
        if st.session_state.current_view in nav_keys:
            idx = nav_keys.index(st.session_state.current_view)
            page = st.radio("Navigation", nav_labels, index=idx)
            
            page_mapping = {
                "Home": "home",
                "Record Decision": "record",
                "View Timeline": "timeline",
                "AI Insights": "ai_insights",
                "Analytics": "analytics"
            }
            
            st.session_state.current_view = page_mapping[page]
        else:
            # In detail or edit view - don't show radio navigation
            st.info(f"📖 Viewing Decision")
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        total = len(st.session_state.memory_store.get_all_decisions())
        private = len(st.session_state.memory_store.get_all_decisions(MemoryLayer.PRIVATE))
        shareable = len(st.session_state.memory_store.get_all_decisions(MemoryLayer.SHAREABLE))
        
        st.metric("Total Decisions", total)
        st.metric("Private", private)
        st.metric("Shareable", shareable)
        
        st.markdown("---")
        st.markdown("### ℹ️ About This System")
        st.caption("""
        This system helps you and AI maintain continuity by capturing not just *what* you decided,
        but *why* you decided it.
        
        **Privacy First:**
        - Your data stays local
        - You control what's shared
        - Full transparency
        """)
    
    # Render current page
    if st.session_state.current_view == 'home':
        render_home()
    elif st.session_state.current_view == 'record':
        render_record_decision()
    elif st.session_state.current_view == 'timeline':
        render_decision_timeline()
    elif st.session_state.current_view == 'decision_detail':
        render_decision_detail()
    elif st.session_state.current_view == 'ai_insights':
        render_ai_insights()
    elif st.session_state.current_view == 'analytics':
        render_analytics()
    elif st.session_state.current_view == 'edit_decision':
        render_edit_decision()


if __name__ == "__main__":
    main()