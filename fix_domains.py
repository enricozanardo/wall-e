#!/usr/bin/env python3
import json

# Load the dataset
print("Loading dataset...")
with open('data/qa_en_dataset.json', 'r') as f:
    data = json.load(f)

# The original domains need to be properly labeled
# Domain 1 = GDPR
# Domain 2 = SDN (already exists)
# Domain 3 = Number Theory
# Domain 4 = Logic
# Domain 5 = Cell Biology
# Domain 6 = Algebra
# Domain 7 = Geometry

# Context descriptions for each domain
context_descriptions = {
    "GDPR": "The GDPR (General Data Protection Regulation) is the European regulation on personal data protection, which came into force on May 25, 2018. The GDPR strengthens the rights of EU citizens regarding privacy and imposes significant obligations on organizations that process personal data. Among its fundamental principles are transparency, purpose limitation, data minimization, accuracy, storage limitation, integrity, and confidentiality. The regulation also introduces the concept of 'accountability' for data controllers and provides for penalties of up to 4% of annual global turnover.",
    
    "SDN": "Software Defined Networks (SDN) are a network architecture that separates the control plane from the data plane, allowing centralized and programmable network management. The SDN controller, such as OpenDaylight or ONOS, communicates with switches via protocols like OpenFlow. SDNs offer advantages such as flexibility, automation, cost reduction, and support for dynamic networks. Common applications include traffic balancing, centralized security, and QoS optimization. Challenges include security, scalability, and interoperability between devices.",
    
    "Number Theory": "Number Theory is the branch of mathematics that studies integers and their properties. It encompasses fundamental concepts such as prime numbers, divisibility, modular arithmetic, and number-theoretic functions. Number theory has ancient origins with contributions from mathematicians like Euclid, Fermat, Euler, and Gauss. Originally considered purely theoretical, modern number theory has found extensive applications in cryptography, computer science, and digital security. Key areas include prime number theory, modular arithmetic, congruence relations, Diophantine equations, and algebraic number theory. The field continues to address significant unsolved problems like the Riemann Hypothesis and Goldbach's Conjecture, while providing the mathematical foundation for secure digital communications.",
    
    "Logic": "Logic and Mathematics are interconnected disciplines studying reasoning, inference, and abstract patterns. Logic provides frameworks for evaluating valid arguments through propositional, predicate, and modal systems. Mathematics extends from arithmetic and algebra to calculus, number theory, and beyond. Together, they form the foundational language of scientific inquiry, offering precise methods to describe relationships, model phenomena, and verify conclusions across disciplines from computer science to physics.",
    
    "Cell Biology": "Cell Biology is the scientific study of the structure, function, and behavior of cells - the fundamental units of life. It explores cell organization, growth, metabolism, and reproduction. Key areas include understanding organelles like mitochondria, ribosomes, and the Golgi apparatus; studying cellular processes like respiration, protein synthesis, and signaling; examining the cell cycle, division mechanisms, and cell specialization; and investigating how cellular components like membranes, cytoskeletons, and genetic material function together. Modern cell biology intersects with molecular biology, genetics, biochemistry, and microscopy to reveal the complex molecular mechanisms that govern cellular life.",
    
    "Algebra": "Algebra is a branch of mathematics dealing with symbols and the rules for manipulating these symbols to represent quantities and their relationships. It encompasses equation solving, the study of abstract structures (groups, rings, fields), and the representation of mathematical patterns. Core concepts include variables, expressions, equations, functions, and systems of equations. Algebra extends from elementary equation solving to advanced topics like abstract algebra, linear algebra, and algebraic geometry, providing tools essential for math, physics, engineering, economics, and computer science.",
    
    "Geometry": "Geometry is the branch of mathematics concerned with the properties and relations of points, lines, surfaces, solids, and higher dimensional analogs. It provides a framework for understanding spatial relationships and forms the foundation for many practical applications. Core areas include Euclidean geometry (studying flat spaces), non-Euclidean geometries (spherical and hyperbolic), analytic geometry (using coordinate systems), differential geometry (examining curved spaces), and topology (focusing on properties preserved under continuous deformations). Applications span from architecture and engineering to computer graphics, navigation systems, and theoretical physics.",
    
    "Ecology": "Ecology is the scientific study of the relationships between living organisms and their physical environment. It examines the distribution, abundance, and interactions of organisms within ecosystems, including energy flow, nutrient cycling, population dynamics, and community structure. Ecological research spans diverse scales from individual behavior to global patterns and addresses pressing environmental challenges such as biodiversity loss, climate change impacts, and ecosystem conservation."
}

# Logic-specific marker questions that should always be identified as Logic
LOGIC_MARKER_QUESTIONS = [
    "what is a syllogism", 
    "what are truth-functional connectives", 
    "what is a tautology in propositional logic",
    "what is a contradiction in propositional logic",
    "what is first-order logic",
    "what is predicate logic",
    "what is modus ponens",
    "what is modus tollens",
    "what is existential quantification",
    "what is universal quantification"
]

# Examine contents of each domain to determine what it really is
def identify_domain(questions, domain_index):
    # Special handling for specific domains based on index
    if domain_index == 3:  # Domain 4 in the original file is almost certainly Logic
        return "Logic"
    
    # Check for marker questions (exact matches)
    for q in questions:
        question = q['question'].lower()
        
        # Check logic questions first with specific markers
        for logic_marker in LOGIC_MARKER_QUESTIONS:
            if logic_marker in question:
                return "Logic"
        
        # Then check other domains
        if "sdn" in question or "openflow" in question:
            return "SDN"
        elif "prime number" in question or "modular arithmetic" in question or "fermat" in question:
            return "Number Theory"
        elif "organelle" in question or "mitochondria" in question or "chromatin" in question:
            return "Cell Biology"
        elif "polynomial" in question or "factoring" in question or "algebra" in question:
            return "Algebra"
        elif "pythagorean" in question or "geometry" in question:
            return "Geometry"
        elif "ecosystem" in question or "ecology" in question or "biodiversity" in question:
            return "Ecology"
    
    # If no exact match found, do keyword frequency analysis
    domain_counts = {"SDN": 0, "Number Theory": 0, "Logic": 0, "Cell Biology": 0, "Algebra": 0, "Geometry": 0, "Ecology": 0}
    
    # Logic-specific keywords
    logic_keywords = ["logic", "syllogism", "proposition", "inference", "truth", "fallacy", "validity", 
                     "premise", "argument", "contradiction", "tautology", "predicate", "quantifier", 
                     "modal", "connective", "deduction", "induction", "entailment", "soundness"]
    
    # Sample the first 10 questions (or all if fewer)
    sample_size = min(10, len(questions))
    for q in questions[:sample_size]:
        content = q['question'].lower() + " " + q['answer'].lower()
        
        # Check for logic keywords with double weight
        for keyword in logic_keywords:
            if keyword in content:
                domain_counts["Logic"] += 2
        
        # Check for keywords for other domains
        if any(kw in content for kw in ["sdn", "network", "openflow", "controller"]):
            domain_counts["SDN"] += 1
            
        if any(kw in content for kw in ["prime", "integer", "modular", "congruence", "theorem"]):
            domain_counts["Number Theory"] += 1
            
        if any(kw in content for kw in ["cell", "organism", "biology", "mitochondria"]):
            domain_counts["Cell Biology"] += 1
            
        if any(kw in content for kw in ["algebra", "polynomial", "equation", "variable"]):
            domain_counts["Algebra"] += 1
            
        if any(kw in content for kw in ["geometry", "area", "volume", "circle", "sphere"]):
            domain_counts["Geometry"] += 1
            
        if any(kw in content for kw in ["ecology", "ecosystem", "species", "habitat"]):
            domain_counts["Ecology"] += 1
    
    # Return domain with highest count
    if max(domain_counts.values()) > 0:
        return max(domain_counts.items(), key=lambda x: x[1])[0]
    else:
        # If still no match, make educated guess based on domain index
        domain_mapping = {
            1: "GDPR",
            2: "SDN",
            3: "Number Theory",
            4: "Logic", 
            5: "Ecology",
            6: "Geometry"
        }
        if domain_index+1 in domain_mapping:
            return domain_mapping[domain_index+1]
        else:
            return "Unknown"

# Store domain types for later reference
identified_domain_types = []

# Build new domains with corrected context descriptions
new_train_data = []

# For each domain, check content and update context description
for i, domain in enumerate(data['train_data']):
    questions = domain['questions']
    original_context = domain['context']
    
    # Check first few sample questions
    print(f"\nDomain {i+1} sample questions:")
    for q in questions[:3]:
        print(f"  - {q['question']}")
    
    # Identify domain type
    if i == 0:
        domain_type = "GDPR"  # Domain 1 is known to be GDPR
    else:
        # For other domains, identify content
        domain_type = identify_domain(questions, i)
    
    identified_domain_types.append(domain_type)
    print(f"Domain {i+1} identified as: {domain_type}")
    
    # Update context description
    if domain_type in context_descriptions:
        updated_context = context_descriptions[domain_type]
    else:
        updated_context = original_context
    
    # Create new domain entry
    new_train_data.append({
        "context": updated_context,
        "questions": questions
    })

# Update the dataset
data['train_data'] = new_train_data

# Save the updated dataset
with open('data/qa_en_dataset_fixed_domains.json', 'w') as f:
    json.dump(data, f, indent=2)

# Print summary
print("\nNew domain structure:")
for i, domain_type in enumerate(identified_domain_types):
    question_count = len(new_train_data[i]['questions'])
    
    # Map domain types to display names
    if domain_type == "Logic":
        display_name = "Logic & Mathematics"
    else:
        display_name = domain_type
        
    print(f"{display_name}: {question_count} questions")

print('\nDataset structure fixed successfully. New file: data/qa_en_dataset_fixed_domains.json') 