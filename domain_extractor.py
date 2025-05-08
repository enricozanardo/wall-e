#!/usr/bin/env python3
import json
import re

# Load the dataset
print("Loading dataset...")
with open('data/qa_en_dataset.json', 'r') as f:
    data = json.load(f)

# Get the first domain's content, which contains mixed domains
gdpr_section = data['train_data'][0]
all_questions = gdpr_section['questions']

# Function to identify domains by characteristic questions
def get_domain_markers():
    markers = {
        'gdpr': [
            "What does GDPR stand for?",
            "What is the right to data portability?",
            "What is the principle of accountability in GDPR?"
        ],
        'sdn': [
            "What is the OpenFlow protocol version history?",
            "How could a government agency implement secure enclaves using SDN?",
            "What is an SDN controller?",
        ],
        'number_theory': [
            "What is the sieve of Eratosthenes?",
            "What is Fermat's Little Theorem?",
            "How could number theory be applied to develop a secure digital signature scheme?"
        ],
        'logic': [
            "What is a syllogism in traditional logic?",
            "What is modus ponens?",
            "What is existential quantification?"
        ],
        'cell_biology': [
            "What is chromatin?",
            "What is the role of the Golgi apparatus?",
            "What are cellular processes?"
        ],
        'algebra': [
            "What is a polynomial?",
            "What is a monomial?",
            "What is factoring in algebra?"
        ],
        'geometry': [
            "What is the Pythagorean theorem?",
            "What is the formula for the area of a circle?",
            "What is the formula for the volume of a sphere?"
        ]
    }
    return markers

# Search for marker questions and identify ranges
def find_domain_ranges():
    markers = get_domain_markers()
    domain_indices = {domain: [] for domain in markers.keys()}
    
    # Find indices of marker questions
    for i, question in enumerate(all_questions):
        q_text = question['question']
        for domain, marker_list in markers.items():
            if q_text in marker_list:
                domain_indices[domain].append(i)
                print(f"Found marker for {domain} at index {i}: {q_text}")
    
    # Identify continuous chunks
    domain_ranges = {}
    for domain, indices in domain_indices.items():
        if indices:
            indices.sort()
            # Look for the first and last index of each domain
            domain_ranges[domain] = (min(indices), max(indices))
            
    # Try to identify boundaries between domains
    sorted_ranges = sorted([(start, end, domain) for domain, (start, end) in domain_ranges.items()], 
                          key=lambda x: x[0])
    
    # Print the identified ranges
    print("\nIdentified domain ranges:")
    for start, end, domain in sorted_ranges:
        print(f"{domain}: {start}-{end} ({end-start+1} questions)")
    
    # Detect domain boundaries
    domain_boundaries = []
    for i in range(len(sorted_ranges)-1):
        current_end = sorted_ranges[i][1]
        next_start = sorted_ranges[i+1][0]
        
        # If domains don't overlap and there's a gap
        if current_end < next_start - 1:
            # Set boundary halfway between
            boundary = (current_end + next_start) // 2
            domain_boundaries.append((sorted_ranges[i][2], sorted_ranges[i+1][2], boundary))
            print(f"Boundary between {sorted_ranges[i][2]} and {sorted_ranges[i+1][2]} at index {boundary}")
        else:
            print(f"WARNING: Possible overlap between {sorted_ranges[i][2]} and {sorted_ranges[i+1][2]}")
    
    return sorted_ranges, domain_boundaries

# Extract specific domain sections
def extract_domain_sections():
    # Find domain ranges
    sorted_ranges, domain_boundaries = find_domain_ranges()
    
    # Determine domain intervals by combining ranges and boundaries
    domain_intervals = []
    
    # Add the first domain starting at 0
    first_domain = sorted_ranges[0][2]
    domain_intervals.append((0, sorted_ranges[0][1], first_domain))
    
    # Use boundaries for middle domains
    for i, (prev_domain, next_domain, boundary) in enumerate(domain_boundaries):
        # End of previous domain is boundary-1
        domain_intervals[-1] = (domain_intervals[-1][0], boundary-1, domain_intervals[-1][2])
        # Start of next domain is boundary
        domain_intervals.append((boundary, sorted_ranges[i+1][1], next_domain))
    
    # Adjust the last domain to extend to the end
    if domain_intervals:
        last_start, last_end, last_domain = domain_intervals[-1]
        domain_intervals[-1] = (last_start, len(all_questions)-1, last_domain)
    
    # Create new domain sections
    domain_sections = {}
    for start, end, domain in domain_intervals:
        domain_sections[domain] = all_questions[start:end+1]
        print(f"\nExtracted {domain} domain with {len(domain_sections[domain])} questions")
        # Show sample questions
        for q in domain_sections[domain][:3]:
            print(f"  - {q['question']}")
    
    return domain_sections

# Reconstruct the dataset with proper domains
def rebuild_dataset():
    domain_sections = extract_domain_sections()
    
    # Create context descriptions
    context_descriptions = {
        'gdpr': "The GDPR (General Data Protection Regulation) is the European regulation on personal data protection, which came into force on May 25, 2018. The GDPR strengthens the rights of EU citizens regarding privacy and imposes significant obligations on organizations that process personal data. Among its fundamental principles are transparency, purpose limitation, data minimization, accuracy, storage limitation, integrity, and confidentiality. The regulation also introduces the concept of 'accountability' for data controllers and provides for penalties of up to 4% of annual global turnover.",
        
        'sdn': "Software Defined Networks (SDN) are a network architecture that separates the control plane from the data plane, allowing centralized and programmable network management. The SDN controller, such as OpenDaylight or ONOS, communicates with switches via protocols like OpenFlow. SDNs offer advantages such as flexibility, automation, cost reduction, and support for dynamic networks. Common applications include traffic balancing, centralized security, and QoS optimization. Challenges include security, scalability, and interoperability between devices.",
        
        'number_theory': "Number Theory is the branch of mathematics that studies integers and their properties. It encompasses fundamental concepts such as prime numbers, divisibility, modular arithmetic, and number-theoretic functions. Number theory has ancient origins with contributions from mathematicians like Euclid, Fermat, Euler, and Gauss. Originally considered purely theoretical, modern number theory has found extensive applications in cryptography, computer science, and digital security. Key areas include prime number theory, modular arithmetic, congruence relations, Diophantine equations, and algebraic number theory. The field continues to address significant unsolved problems like the Riemann Hypothesis and Goldbach's Conjecture, while providing the mathematical foundation for secure digital communications.",
        
        'logic': "Logic and Mathematics are interconnected disciplines studying reasoning, inference, and abstract patterns. Logic provides frameworks for evaluating valid arguments through propositional, predicate, and modal systems. Mathematics extends from arithmetic and algebra to calculus, number theory, and beyond. Together, they form the foundational language of scientific inquiry, offering precise methods to describe relationships, model phenomena, and verify conclusions across disciplines from computer science to physics.",
        
        'cell_biology': "Cell Biology is the scientific study of the structure, function, and behavior of cells - the fundamental units of life. It explores cell organization, growth, metabolism, and reproduction. Key areas include understanding organelles like mitochondria, ribosomes, and the Golgi apparatus; studying cellular processes like respiration, protein synthesis, and signaling; examining the cell cycle, division mechanisms, and cell specialization; and investigating how cellular components like membranes, cytoskeletons, and genetic material function together. Modern cell biology intersects with molecular biology, genetics, biochemistry, and microscopy to reveal the complex molecular mechanisms that govern cellular life.",
        
        'algebra': "Algebra is a branch of mathematics dealing with symbols and the rules for manipulating these symbols to represent quantities and their relationships. It encompasses equation solving, the study of abstract structures (groups, rings, fields), and the representation of mathematical patterns. Core concepts include variables, expressions, equations, functions, and systems of equations. Algebra extends from elementary equation solving to advanced topics like abstract algebra, linear algebra, and algebraic geometry, providing tools essential for math, physics, engineering, economics, and computer science.",
        
        'geometry': "Geometry is the branch of mathematics concerned with the properties and relations of points, lines, surfaces, solids, and higher dimensional analogs. It provides a framework for understanding spatial relationships and forms the foundation for many practical applications. Core areas include Euclidean geometry (studying flat spaces), non-Euclidean geometries (spherical and hyperbolic), analytic geometry (using coordinate systems), differential geometry (examining curved spaces), and topology (focusing on properties preserved under continuous deformations). Applications span from architecture and engineering to computer graphics, navigation systems, and theoretical physics."
    }
    
    # Build new domain sections
    new_train_data = []
    
    # Add domains in logical order
    domain_order = ['gdpr', 'sdn', 'number_theory', 'logic', 'cell_biology', 'algebra', 'geometry']
    
    for domain in domain_order:
        if domain in domain_sections and domain_sections[domain]:
            new_train_data.append({
                "context": context_descriptions[domain],
                "questions": domain_sections[domain]
            })
    
    # Add existing domains from original data (ecology, etc.)
    print("\nAdding original domains:")
    for i in range(1, len(data['train_data'])):
        context_text = data['train_data'][i]['context']
        question_count = len(data['train_data'][i]['questions'])
        
        # Try to identify the domain by its context
        domain_name = f"Original Domain {i+1}"
        if 'ecology' in context_text.lower():
            domain_name = "Ecology"
        elif 'geometry' in context_text.lower():
            domain_name = "Additional Geometry"
        
        print(f"Adding {domain_name} with {question_count} questions")
        new_train_data.append(data['train_data'][i])
    
    data['train_data'] = new_train_data
    
    # Save the updated dataset
    with open('data/qa_en_dataset_extracted.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    # Print summary
    print("\nNew domain structure:")
    for i, context in enumerate(new_train_data):
        # Try to identify domain name
        context_text = context['context'].lower()
        domain_name = f"Domain {i+1}"
        
        if 'gdpr' in context_text:
            domain_name = "GDPR"
        elif 'software defined networks' in context_text:
            domain_name = "SDN"
        elif 'number theory' in context_text:
            domain_name = "Number Theory"
        elif 'logic' in context_text:
            domain_name = "Logic & Mathematics"
        elif 'cell biology' in context_text:
            domain_name = "Cell Biology"
        elif 'algebra' in context_text:
            domain_name = "Algebra"
        elif 'geometry' in context_text and i < 7:
            domain_name = "Geometry"
        elif 'ecology' in context_text:
            domain_name = "Ecology"
        
        print(f"{domain_name}: {len(context['questions'])} questions")
    
    print('\nDataset structure fixed successfully. New file: data/qa_en_dataset_extracted.json')

# Run the extraction process
if __name__ == "__main__":
    rebuild_dataset() 