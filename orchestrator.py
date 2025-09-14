GROQ_API_KEY = ''
import json
import random
import os
import argparse
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal
from datetime import datetime
from pathlib import Path
import pandas as pd

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage,SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

llm = ChatOpenAI(model = "qwen/qwen3-32b", api_key=GROQ_API_KEY, temperature=0.3, base_url='https://api.groq.com/openai/v1')

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

class WorkflowState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: Dict[str, Any]
    target_metadata: Optional[Dict[str, Any]]
    library: Optional[List[Dict[str, Any]]]
    docking_results: Optional[List[Dict[str, Any]]]
    top_hits: Optional[List[Dict[str, Any]]]
    summary: Optional[str]
    knowledge_response: Optional[str]
    workflow_type: str
    llm_decisions: Dict[str, Any]
    memory_context: Dict[str, Any]

# Agent Classes
class TargetParserAgent:
    """Target Parser Agent - validates and standardizes protein targets"""
    
    def __init__(self):
        self.protein_mapping = {
            "egfr": "1A4G",
            "p53": "1TUP", 
            "bcl2": "1G5M",
            "hdac": "1C3S",
            "cdk2": "1HCK",
            "vegfr": "3VHE"
        }
    
    def process(self, target: str) -> Dict[str, Any]:
        """Parse the protien name and validate it """
        if not target:
            return {"target_id": None, "chain": "A", "valid": False, "error": "No target provided"}
            
        target_clean = target.strip().upper()
        target_lower = target.lower()
        
        # Check if it's a protein name first
        if target_lower in self.protein_mapping:
            return {
                "target_id": self.protein_mapping[target_lower],
                "chain": "A",
                "source": "protein_name",
                "original_name": target,
                "valid": True
            }

        # mock Checking the length 
        if len(target_clean) == 4 and target_clean.isalnum():
            return {
                "target_id": target_clean,
                "chain": "A", 
                "source": "pdb_id",
                "original_name": target,
                "valid": True
            }

        return {
            "target_id": None,
            "chain": "A",
            "valid": False,
            "error": f"Unknown target: {target}"
        }

class LibraryGeneratorAgent:
    """Library Generator Agent - generates mock molecular libraries"""
    
    def __init__(self):
        self.base_smiles = [
            "CCO",  
            "c1ccccc1",  
            "CC(C)O",  
            "CCCC",  
            "c1ccc(cc1)O",  
            "CCN",  
            "CC(=O)O",  
            "c1ccc2c(c1)cccc2",  
            "CC(C)(C)O",  
            "c1ccc(cc1)N",  
            "CCCCO",
            "c1ccc(cc1)C",  
            "CCO[CH2]",  
            "CC(C)C",  
            "c1ccc(cc1)Cl"  
        ]
    
    def generate_library(self, size: int) -> List[Dict[str, Any]]:
        """Generate molecular library with SMILES strings"""

        library = []
        
        for i in range(size):
            if i < len(self.base_smiles):
                smiles = self.base_smiles[i]
            else:
                
                base = random.choice(self.base_smiles)
                smiles = self._modify_smiles(base, i)
            
            library.append({
                "ligand_id": f"L{i+1}", 
                "smiles": smiles
            })
        
        return library
    
    def _modify_smiles(self, base_smiles: str, seed: int) -> str:
        """Simple SMILES modifications for variety"""
        random.seed(seed)  
        modifications = [
            lambda s: s, 
            lambda s: s.replace("C", "CC") if "C" in s else s,
            lambda s: s + "O" if not s.endswith("O") else s,
            lambda s: "N" + s if not s.startswith("N") else s,
        ]
        return random.choice(modifications)(base_smiles)

class MockDockingAgent:
    """Mock Docking Agent - performs deterministic molecular docking simulation"""
    
    def dock(self, target_metadata: Dict[str, Any], library: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform mock docking with deterministic scoring
        Uses formula from PDF: score = -(hash(smiles) % 7 + 4) for range -4.0 to -10.0
        """
        results = []
        
        for ligand in library:
            smiles = ligand["smiles"]
            
            base_score = -(hash(smiles) % 7 + 4)  # -4 to -10

            decimal_offset = (hash(smiles) % 10) / 10.0  

            score = base_score - decimal_offset 
 
            
            results.append({
                "ligand_id": ligand["ligand_id"],
                "smiles": smiles,
                "docking_score": float(score),
                "target": target_metadata["target_id"]
            })
        
        return results

class ScoringRankingAgent:

    """ranks and selects top hits"""
    
    def rank_hits(self, docking_results: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:

        """Rank ligands by docking score and select top hits"""

        sorted_results = sorted(docking_results, key=lambda x: x["docking_score"])
        
        # Add rank and select top hits - format as per PDF
        top_hits = []
        for i, hit in enumerate(sorted_results[:top_n]):
            hit_with_rank = hit.copy()
            hit_with_rank["rank"] = i + 1
            top_hits.append(hit_with_rank)
        
        return top_hits

class SummaryWriterAgent:

    """Summary Writer Agent - generates screening summaries"""
    
    def generate_summary(self, target_metadata: Dict[str, Any], 
                        library_size: int, top_hits: List[Dict[str, Any]]) -> str:
        
        """Generate Markdown summary"""

        
        target_name = target_metadata.get('original_name', target_metadata.get('target_id', 'Unknown'))
        target_id = target_metadata.get('target_id', 'Unknown')
        
        summary = f""" ## Virtual Screening Summary
#### target inofrmation
 - Targetname: {target_name} 
 - target_id : {target_id}

Library Size: {library_size} molecules
Method: Mock docking simulation
Top Hits:
"""
        
        for hit in top_hits:
            summary += f"{hit['rank']}. {hit['ligand_id']} – {hit['smiles']} (Docking Score: {hit['docking_score']})\n"
        summary += f"""
#### Statistics
- Best Docking Score: {top_hits[0]['docking_score']} kcal/mol
- Score Range: {top_hits[-1]['docking_score']} to {top_hits[0]['docking_score']} kcal/mol

"""

        summary += "Recommendation: Proceed with experimental validation.\n"

        return summary

class KnowledgeAgent:
    """Knowledge Agent - provides chemistry and pharmaceutical knowledge"""

    def __init__(self, knowledge_file: str = "knowledge_base.json"):
        # Loading json file
        if os.path.exists(knowledge_file):
            with open(knowledge_file, "r") as f:
                self.knowledge_base = json.load(f)
        else:
            self.knowledge_base = {}
            

    def query(self, question: str) -> str:

        """Answer chemistry and pharmaceutical questions"""
        question_lower = question.lower().strip()

        # Check for exact matches and partial matches
        for key, info in self.knowledge_base.items():
            if key in question_lower:
                # Just dump JSON content in a readable way
                return json.dumps(info, indent=2)

        return f"No information found for '{question}'. Try asking about: {', '.join(self.knowledge_base.keys())}"
    
class MemoryModule:
    """Memory Module - handles memory session"""
    
    def __init__(self, memory_file: str = "session_memory.json"):
        self.memory_file = memory_file
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from file"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {"last_target": None, "last_library_size": None, "history": []}
    
    def save_memory(self):
        """Save memory to file"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def update_session_context(self, query: Dict[str, Any], result: Dict[str, Any]):
        """Update session memory - store last target and library size"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result_summary": {
                "workflow_type": result.get("workflow_type"),
                "success": result.get("success", True)
            }
        }
        
        self.memory["history"].append(entry)
        
        if "target" in query:
            self.memory["last_target"] = query["target"]
        if "library_size" in query:
            self.memory["last_library_size"] = query["library_size"]

        if len(self.memory["history"]) > 5:
            self.memory["history"] = self.memory["history"][-5:]
        
        self.save_memory()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current memory context"""
        return {
            "last_target": self.memory.get("last_target"),
            "last_library_size": self.memory.get("last_library_size"),
            "session_count": len(self.memory.get("history", []))
        }
    
    def apply_memory_to_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Apply memory context to current query using LLM decision"""
        memory_context = self.get_context()
        
        # Use LLM to decide if memory should be applied
        if memory_context["last_target"] and "target" not in query:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are helping decide whether to use remembered context for a query.
The user previously used target: {last_target}
Current query: {current_query}

Should we use the remembered target? Consider:
- If query mentions screening/molecules/docking but no target → likely YES
- If query is completely different topic → NO
- If query explicitly mentions different target → NO

Return only: {{"use_last_target": true/false}}"""),
                ("human", f"Last target: {memory_context['last_target']}, Current query: {json.dumps(query)}")
            ])
            
            try:
                response = llm.invoke(prompt.format_messages())
                decision = json.loads(response.content)
                if decision.get("use_last_target", False):
                    query["target"] = memory_context["last_target"]
                    print(f"Memory: Using remembered target '{memory_context['last_target']}'")
            except Exception:
                if ("library_size" in query or 
                    any(word in str(query).lower() for word in ["screen", "dock", "molecules", "compounds"])):
                    query["target"] = memory_context["last_target"]
                    print(f"Memory: Using remembered target '{memory_context['last_target']}'")
        
        return query

class VirtualScreeningOrchestrator:
    """Main Orchestrator Agent - as per PDF specifications"""
    
    def __init__(self):
        self.llm = llm
        self.memory = MemoryModule()
        self.target_parser = TargetParserAgent()
        self.library_generator = LibraryGeneratorAgent()
        self.docking_agent = MockDockingAgent()
        self.scoring_agent = ScoringRankingAgent()
        self.summary_writer = SummaryWriterAgent()
        self.knowledge_agent = KnowledgeAgent()
        
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """Build LangGraph workflow with adaptive routing"""
        workflow = StateGraph(WorkflowState)
        
       
        workflow.add_node("router", self._router_node)
        workflow.add_node("memory_manager", self._memory_node)
        workflow.add_node("target_parser", self._target_parser_node)
        workflow.add_node("library_decision", self._library_decision_node)
        workflow.add_node("library_generator", self._library_generator_node)
        workflow.add_node("docking", self._docking_node)
        workflow.add_node("ranking", self._ranking_node)
        workflow.add_node("summary_decision", self._summary_decision_node)
        workflow.add_node("summary", self._summary_node)
        workflow.add_node("knowledge", self._knowledge_node)
        workflow.add_node("output", self._output_node)
        
    
        workflow.set_entry_point("router")
        
       
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {"knowledge": "knowledge", "screening": "memory_manager"}
        )
        
        
        workflow.add_edge("knowledge", "output")
        
      
        workflow.add_edge("memory_manager", "target_parser")
        workflow.add_edge("target_parser", "library_decision")
        
       
        workflow.add_conditional_edges(
            "library_decision",
            self._library_choice,
            {"generate": "library_generator", "skip": "docking"}
        )
        
        workflow.add_edge("library_generator", "docking")
        workflow.add_edge("docking", "ranking")
        workflow.add_edge("ranking", "summary_decision")
        
        
        workflow.add_conditional_edges(
            "summary_decision",
            self._summary_choice,
            {"generate": "summary", "skip": "output"}
        )
        
        workflow.add_edge("summary", "output")
        workflow.add_edge("output", END)
        
        return workflow
    
    def _router_node(self, state: WorkflowState) -> WorkflowState:

        """LLM-powered routing decision"""
        query = state["query"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the query and determine the workflow type:
1. "knowledge" - Questions about chemistry, drug discovery, explanations (e.g., "What is ADMET?")
2. "screening" - Requests to perform virtual screening (e.g., {"target": "EGFR", "library_size": 10})

Return only: {"workflow_type": "knowledge" or "screening"}"""),
            ("human", f"Query: {json.dumps(query)}")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            decision = json.loads(response.content)
            workflow_type = decision.get("workflow_type", "screening")
        except:
            # Fallback logic
            if (isinstance(query, str) or 
                (isinstance(query, dict) and "question" in query) or
                (isinstance(query, str) and any(word in query.lower() for word in ["what", "explain", "define"]))):
                workflow_type = "knowledge"
            else:
                workflow_type = "screening"
        
        state["workflow_type"] = workflow_type
        state["llm_decisions"] = {"router": workflow_type}
        
        return state
    
    def _route_decision(self, state: WorkflowState) -> Literal["knowledge", "screening"]:
        return state["workflow_type"]
    
    def _memory_node(self, state: WorkflowState) -> WorkflowState:
        """Apply memory context"""
        query = state["query"]
        
        # Apply memory using LLM decision
        query = self.memory.apply_memory_to_query(query)
        
        state["query"] = query
        state["memory_context"] = self.memory.get_context()
        
        return state
    
    def _target_parser_node(self, state: WorkflowState) -> WorkflowState:
        """Parse and validate target"""
        query = state["query"]
        target = query.get("target")
        
        if not target:
            state["target_metadata"] = {"valid": False, "error": "No target specified"}
            return state
        
        result = self.target_parser.process(target)
        state["target_metadata"] = result
        
        return state
    
    def _library_decision_node(self, state: WorkflowState) -> WorkflowState:
        """LLM decides whether to generate library or skip"""
        query = state["query"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Decide library generation strategy:
- Skip if: smiles_file provided, custom library specified
- Generate if: standard screening requested
Return: {"action": "generate" or "skip"}"""),
            ("human", f"Query: {json.dumps(query)}")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            decision = json.loads(response.content)
            action = decision.get("action", "generate")
        except:
           
            if "smiles_file" in query or "custom_library" in query:
                action = "skip"
            else:
                action = "generate"
        
        state["llm_decisions"]["library"] = action
        
        # If skipping, this is mock library
        if action == "skip":
            state["library"] = [
                {"ligand_id": "L1", "smiles": "CCO"},
                {"ligand_id": "L2", "smiles": "c1ccccc1"},
                {"ligand_id": "L3", "smiles": "CC(C)O"}
            ]
        
        return state
    
    def _library_choice(self, state: WorkflowState) -> Literal["generate", "skip"]:
        return state["llm_decisions"]["library"]
    
    def _library_generator_node(self, state: WorkflowState) -> WorkflowState:
        """Generate molecular library"""
        query = state["query"]
        library_size = query.get("library_size", 10)
        
        library = self.library_generator.generate_library(library_size)
        state["library"] = library
        
        return state
    
    def _docking_node(self, state: WorkflowState) -> WorkflowState:
        """Perform mock docking"""
        target_metadata = state["target_metadata"]
        library = state["library"]
        
        if not target_metadata["valid"]:
            state["docking_results"] = []
            return state
        
        results = self.docking_agent.dock(target_metadata, library)
        state["docking_results"] = results
        
        return state
    
    def _ranking_node(self, state: WorkflowState) -> WorkflowState:
        """Rank and select top hits"""
        docking_results = state["docking_results"]
        top_n = state["query"].get("top_n", 5)  
        
        top_hits = self.scoring_agent.rank_hits(docking_results, top_n)
        state["top_hits"] = top_hits
        
        return state
    
    def _summary_decision_node(self, state: WorkflowState) -> WorkflowState:
        """LLM decides on summary generation"""
        query = state["query"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Decide summary generation:
- Skip if: skip_summary=true
- Generate otherwise
Return: {"action": "generate" or "skip"}"""),
            ("human", f"Query: {json.dumps(query)}")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            decision = json.loads(response.content)
            action = decision.get("action", "generate")
        except:
            action = "skip" if query.get("skip_summary", False) else "generate"
        
        state["llm_decisions"]["summary"] = action
        
        return state
    
    def _summary_choice(self, state: WorkflowState) -> Literal["generate", "skip"]:
        return state["llm_decisions"]["summary"]
    
    def _summary_node(self, state: WorkflowState) -> WorkflowState:
        """Generate summary"""
        target_metadata = state["target_metadata"]
        library = state["library"]
        top_hits = state["top_hits"]
        
        summary = self.summary_writer.generate_summary(target_metadata, len(library), top_hits)
        state["summary"] = summary
        
        return state
    
    def _knowledge_node(self, state: WorkflowState) -> WorkflowState:
        """Handle knowledge queries"""
        query = state["query"]
        
        if isinstance(query, str):
            question = query
        else:
            question = query.get("question", str(query))
        
        answer = self.knowledge_agent.query(question)
        state["knowledge_response"] = answer
        
        return state
    
    def _output_node(self, state: WorkflowState) -> WorkflowState:
        """Generate output files as per PDF format"""
        workflow_type = state["workflow_type"]
        
        if workflow_type == "screening":

            # Saving docking results CSV
            if state.get("docking_results"):
                df = pd.DataFrame(state["docking_results"])
                df.to_csv(self.output_dir / "docking_results.csv", index=False)
            
           
            if state.get("top_hits"):
                df = pd.DataFrame(state["top_hits"])
                
                column_order = ["rank", "ligand_id", "smiles", "docking_score"]
                df = df[column_order]
                df.columns = ["Rank", "Ligand_id", "SMILES", "Docking_score"]  # PDF format
                df.to_csv(self.output_dir / "top_hits.csv", index=False)
            
           
            if state.get("summary"):
                with open(self.output_dir / "summary.md", 'w') as f:
                    f.write(state["summary"])
            
           
            query = state["query"]
            result = {
                "workflow_type": workflow_type,
                "target": state.get("target_metadata"),
                "hits_count": len(state.get("top_hits", [])),
                "success": state.get("target_metadata", {}).get("valid", False)
            }
            self.memory.update_session_context(query, result)
        
        return state
    
    def process_query(self, query) -> Dict[str, Any]:
        """Process query through workflow"""
        
        if isinstance(query, str):
            query_dict = {"question": query}
        else:
            query_dict = query
            
        initial_state = WorkflowState(
            messages=[HumanMessage(content=f"Process: {query}")],
            query=query_dict,
            target_metadata=None,
            library=None,
            docking_results=None,
            top_hits=None,
            summary=None,
            knowledge_response=None,
            workflow_type="",
            llm_decisions={},
            memory_context={}
        )
        
        app = self.workflow.compile()
        final_state = app.invoke(initial_state)
        
        return {
            "workflow_type": final_state["workflow_type"],
            "llm_decisions": final_state["llm_decisions"],
            "target": final_state.get("target_metadata"),
            "hits": len(final_state.get("top_hits", []) or []),
            "knowledge_response": final_state.get("knowledge_response"),
            "success": final_state.get("target_metadata", {}).get("valid", True) if final_state["workflow_type"] == "screening" else True
        }

def main():
    """CLI entrypoint as per PDF requirements"""
    parser = argparse.ArgumentParser(description="Virtual Screening Orchestrator")
    parser.add_argument("--query", required=True, help="Path to query JSON file")
    
    args = parser.parse_args()
    
  
    try:
        with open(args.query, 'r') as f:
            query = json.load(f)
    except Exception as e:
        print(f"Error loading query file: {e}")
        return
    
    
    orchestrator = VirtualScreeningOrchestrator()
    result = orchestrator.process_query(query)
    
    print(f"Workflow completed: {result['workflow_type']}")
    print(f"Outputs generated in outputs/ directory")
    
    if result['workflow_type'] == 'screening':
        target_info = result.get('target', {})
        if target_info and target_info.get('valid'):
            print(f"Target: {target_info.get('target_id', 'Unknown')}")
            print(f"Top hits: {result['hits']}")
        else:
            print(f"Error: {target_info.get('error', 'Invalid target')}")
    elif result['workflow_type'] == 'knowledge':
        print("Knowledge response:")
        print(result.get("knowledge_response", "No response"))

if __name__ == "__main__":
    main()