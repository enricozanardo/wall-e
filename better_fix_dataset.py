#!/usr/bin/env python3
import json
import re

# Load the dataset
print("Loading dataset...")
with open('data/qa_en_dataset.json', 'r') as f:
    data = json.load(f)

# First section is GDPR
gdpr_section = data['train_data'][0]
all_questions = gdpr_section['questions']

# More comprehensive keywords for domain detection
domain_keywords = {
    'gdpr': ['gdpr', 'data protection', 'privacy', 'personal data', 'data subject', 'controller', 'processor', 'compliance', 'consent', 'regulation', 'supervisory authority', 'data breach'],
    
    'sdn': ['sdn', 'software defined network', 'openflow', 'controller', 'switch', 'packet', 'protocol', 'traffic', 'network', 'routing', 'northbound', 'southbound', 'flow table', 'network slice', 'virtualization'],
    
    'number_theory': ['prime', 'number theory', 'integer', 'theorem', 'modular', 'arithmetic', 'cryptography', 'divisor', 'congruence', 'factor', 'gcd', 'euler', 'fermat', 'fibonacci', 'diophantine', 'prime factorization', 'number field', 'algebraic number'],
    
    'logic': ['logic', 'syllogism', 'proposition', 'deduction', 'inference', 'truth', 'fallacy', 'validity', 'premise', 'argument', 'contradiction', 'tautology', 'modus ponens', 'modus tollens', 'quantifier', 'existential', 'universal', 'modal logic', 'predicate logic'],
    
    'cell_biology': ['cell', 'biology', 'organelle', 'mitochondria', 'dna', 'rna', 'chromosome', 'protein', 'enzyme', 'membrane', 'nucleus', 'cytoplasm', 'golgi', 'ribosome', 'cell cycle', 'endoplasmic reticulum', 'lysosome', 'vesicle', 'cellular', 'chromatin', 'histones', 'apoptosis', 'senescence', 'autophagy', 'cytoskeleton', 'flagella', 'cilia', 'stem cell', 'differentiation', 'transcription', 'translation'],
    
    'algebra': ['polynomial', 'equation', 'variable', 'algebra', 'matrix', 'linear', 'vector', 'function', 'coefficient', 'expression', 'root', 'quadratic', 'cubic', 'eigenvalue', 'eigenvector', 'determinant', 'logarithm', 'exponent', 'monomial', 'binomial', 'trinomial', 'identity'],
    
    'geometry': ['geometry', 'sphere', 'circle', 'triangle', 'polyhedron', 'angle', 'volume', 'area', 'cylinder', 'cone', 'euclidean', 'pythagorean', 'polygon', 'surface', 'cube', 'rectangular', 'prism', 'circumference', 'diameter', 'radius', 'hypotenuse', 'parallelogram', 'isosceles', 'trapezoid', 'coordinate', 'projection', 'transformation', 'rotation', 'translation']
}

# Print a sample of questions to debug
print("Sampling some questions:")
for i in range(0, len(all_questions), len(all_questions) // 10):
    if i < len(all_questions):
        print(f"  {i}: {all_questions[i]['question']}")

# Assign domains based on keyword matching with improved scoring
def classify_question(question):
    q_text = question['question'].lower()
    a_text = question['answer'].lower()
    full_text = q_text + ' ' + a_text
    
    # Count keyword matches for each domain with weighted scoring
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        # Question title matches are worth more
        title_score = sum(3 for keyword in keywords if keyword in q_text)
        # Full text matches
        full_score = sum(1 for keyword in keywords if keyword in full_text)
        domain_scores[domain] = title_score + full_score
    
    # Check for specific biology patterns
    if any(term in full_text for term in ['cell', 'organelle', 'dna', 'rna', 'chromatin', 'protein']):
        domain_scores['cell_biology'] += 10
        
    # Check for specific geometry patterns
    if re.search(r'(area|volume|surface) of a (sphere|cube|cylinder|cone|triangle|circle)', full_text):
        domain_scores['geometry'] += 10
        
    # Check for specific logic patterns
    if re.search(r'(syllogism|deduction|inference|truth|fallacy|premise|argument)', full_text):
        domain_scores['logic'] += 10

    # Assign to domain with highest score
    if max(domain_scores.values()) > 0:
        assigned_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
    else:
        assigned_domain = 'unknown'
        
    return assigned_domain

# Classify all questions
print("Classifying questions...")
question_domains = [classify_question(q) for q in all_questions]

# Print domain statistics
domains_count = {}
for domain in question_domains:
    domains_count[domain] = domains_count.get(domain, 0) + 1
    
print("\nDomain distribution in original GDPR section:")
for domain, count in sorted(domains_count.items(), key=lambda x: x[1], reverse=True):
    print(f"  {domain}: {count} questions")

# Create new context sections for each domain
gdpr_questions = []
sdn_questions = []
number_theory_questions = []
logic_questions = []
cell_biology_questions = []
algebra_questions = []
geometry_questions = []
unknown_questions = []

# Group questions by domain
for i, domain in enumerate(question_domains):
    question = all_questions[i]
    if domain == 'gdpr':
        gdpr_questions.append(question)
    elif domain == 'sdn':
        sdn_questions.append(question)
    elif domain == 'number_theory':
        number_theory_questions.append(question)
    elif domain == 'logic':
        logic_questions.append(question)
    elif domain == 'cell_biology':
        cell_biology_questions.append(question)
    elif domain == 'algebra':
        algebra_questions.append(question)
    elif domain == 'geometry':
        geometry_questions.append(question)
    else:
        # For unknown, look at the original context section (Domain 1 = GDPR)
        # If it's in lines 1-202, it's probably genuinely GDPR
        if i < 202:  
            gdpr_questions.append(question)
        else:
            unknown_questions.append(question)

# Print sample categorizations
print("\nSample categorizations:")
domains_to_check = ['cell_biology', 'logic', 'number_theory', 'algebra', 'geometry']
for domain in domains_to_check:
    questions = []
    if domain == 'gdpr':
        questions = gdpr_questions
    elif domain == 'sdn':
        questions = sdn_questions
    elif domain == 'number_theory':
        questions = number_theory_questions
    elif domain == 'logic':
        questions = logic_questions
    elif domain == 'cell_biology':
        questions = cell_biology_questions
    elif domain == 'algebra':
        questions = algebra_questions
    elif domain == 'geometry':
        questions = geometry_questions
    
    print(f"\n{domain.upper()} ({len(questions)} questions):")
    for q in questions[:3]:  # Show first 3
        print(f"  - {q['question']}")

# Update GDPR section
gdpr_section['questions'] = gdpr_questions

# Create new context sections
new_contexts = [
    {
        'context': 'Software Defined Networks (SDN) are a network architecture that separates the control plane from the data plane, allowing centralized and programmable network management. The SDN controller, such as OpenDaylight or ONOS, communicates with switches via protocols like OpenFlow. SDNs offer advantages such as flexibility, automation, cost reduction, and support for dynamic networks. Common applications include traffic balancing, centralized security, and QoS optimization. Challenges include security, scalability, and interoperability between devices.',
        'questions': sdn_questions
    },
    {
        'context': 'Number Theory is the branch of mathematics that studies integers and their properties. It encompasses fundamental concepts such as prime numbers, divisibility, modular arithmetic, and number-theoretic functions. Number theory has ancient origins with contributions from mathematicians like Euclid, Fermat, Euler, and Gauss. Originally considered purely theoretical, modern number theory has found extensive applications in cryptography, computer science, and digital security. Key areas include prime number theory, modular arithmetic, congruence relations, Diophantine equations, and algebraic number theory. The field continues to address significant unsolved problems like the Riemann Hypothesis and Goldbach\'s Conjecture, while providing the mathematical foundation for secure digital communications.',
        'questions': number_theory_questions
    },
    {
        'context': 'Cell Biology is the scientific study of the structure, function, and behavior of cells - the fundamental units of life. It explores cell organization, growth, metabolism, and reproduction. Key areas include understanding organelles like mitochondria, ribosomes, and the Golgi apparatus; studying cellular processes like respiration, protein synthesis, and signaling; examining the cell cycle, division mechanisms, and cell specialization; and investigating how cellular components like membranes, cytoskeletons, and genetic material function together. Modern cell biology intersects with molecular biology, genetics, biochemistry, and microscopy to reveal the complex molecular mechanisms that govern cellular life.',
        'questions': cell_biology_questions
    },
    {
        'context': 'Logic and Mathematics encompass formal systems for reasoning and abstract pattern analysis. Logic provides tools for evaluating valid arguments through propositional, predicate, and modal frameworks. Mathematics extends from arithmetic and algebra to advanced fields like calculus, number theory, and topology. Together, they form the language of scientific inquiry, offering precise methods to describe relationships, model phenomena, and verify conclusions. These interconnected disciplines underpin fields from computer science and physics to economics and engineering, providing both practical tools and avenues for exploring abstract truths.',
        'questions': logic_questions + algebra_questions
    }
]

# Add a section for geometry if it contains questions
if geometry_questions:
    new_contexts.append({
        'context': 'Geometry is the branch of mathematics concerned with the properties and relations of points, lines, surfaces, solids, and higher dimensional analogs. It provides a framework for understanding spatial relationships and forms the foundation for many practical applications. Core areas include Euclidean geometry (studying flat spaces), non-Euclidean geometries (spherical and hyperbolic), analytic geometry (using coordinate systems), differential geometry (examining curved spaces), and topology (focusing on properties preserved under continuous deformations). Applications span from architecture and engineering to computer graphics, navigation systems, and theoretical physics.',
        'questions': geometry_questions
    })

# Handle unknown questions if any
if unknown_questions:
    print(f"\nWarning: {len(unknown_questions)} questions couldn't be categorized!")
    for q in unknown_questions[:5]:
        print(f"  - {q['question']}")
    
    # Manual review and confirm where to place these
    print("\nUnknown questions will be placed in a separate section for manual review.")
    new_contexts.append({
        'context': 'Miscellaneous questions requiring manual categorization.',
        'questions': unknown_questions
    })

# Build new train_data structure
new_train_data = [gdpr_section]
for ctx in new_contexts:
    if len(ctx['questions']) > 0:  # Only add non-empty sections
        new_train_data.append(ctx)

# Add existing domains from original data (ecology, etc.)
original_domains_added = 0
for i in range(1, len(data['train_data'])):
    domain_name = f"Original Domain {i+1}"
    context_text = data['train_data'][i]['context']
    
    # Try to identify the domain by its context
    if 'ecology' in context_text.lower():
        domain_name = "Ecology"
    elif 'geometry' in context_text.lower():
        domain_name = "Additional Geometry"
    
    print(f"Adding {domain_name} with {len(data['train_data'][i]['questions'])} questions")
    new_train_data.append(data['train_data'][i])
    original_domains_added += 1

data['train_data'] = new_train_data

# Save the updated dataset
with open('data/qa_en_dataset_better_fixed.json', 'w') as f:
    json.dump(data, f, indent=2)

# Print summary
print("\nNew domain structure:")
for i, context in enumerate(new_train_data):
    domain_name = f"Domain {i+1}"
    if i == 0:
        domain_name = "GDPR"
    elif i == 1:
        domain_name = "SDN"
    elif i == 2:
        domain_name = "Number Theory"
    elif i == 3:
        domain_name = "Cell Biology"
    elif i == 4:
        domain_name = "Logic & Mathematics"
    elif i == 5:
        domain_name = "Geometry"
    elif i == 6:
        domain_name = "Unknown/Miscellaneous"
    
    print(f"{domain_name}: {len(context['questions'])} questions")
    
print('\nDataset structure fixed successfully. New file: data/qa_en_dataset_better_fixed.json') 