#!/usr/bin/env python3
import json

# Load the dataset
with open('data/qa_en_dataset.json', 'r') as f:
    data = json.load(f)

# First section is GDPR
gdpr_section = data['train_data'][0]
all_questions = gdpr_section['questions']

# Index positions to split at (determined by analysis of the file)
sdn_start_index = None
number_theory_start_index = None
logic_start_index = None

# Find the indexes
for i, question in enumerate(all_questions):
    if question['question'] == 'How could a government agency implement secure enclaves using SDN?':
        sdn_start_index = i
    elif question['question'] == 'How could number theory be applied to develop a secure digital signature scheme?':
        number_theory_start_index = i
    elif question['question'] == 'What is a syllogism in traditional logic?':
        logic_start_index = i
    
    # Print for debugging
    if sdn_start_index and number_theory_start_index and logic_start_index:
        print(f"Found split points - SDN: {sdn_start_index}, Number Theory: {number_theory_start_index}, Logic: {logic_start_index}")
        break

# Split questions into domains
gdpr_questions = all_questions[:sdn_start_index]
sdn_questions = all_questions[sdn_start_index:number_theory_start_index]
number_theory_questions = all_questions[number_theory_start_index:logic_start_index]
logic_questions = all_questions[logic_start_index:]

# Create context sections
gdpr_section['questions'] = gdpr_questions

sdn_context = {
    'context': 'Software Defined Networks (SDN) are a network architecture that separates the control plane from the data plane, allowing centralized and programmable network management. The SDN controller, such as OpenDaylight or ONOS, communicates with switches via protocols like OpenFlow. SDNs offer advantages such as flexibility, automation, cost reduction, and support for dynamic networks. Common applications include traffic balancing, centralized security, and QoS optimization. Challenges include security, scalability, and interoperability between devices.',
    'questions': sdn_questions
}

number_theory_context = {
    'context': 'Number Theory is the branch of mathematics that studies integers and their properties. It encompasses fundamental concepts such as prime numbers, divisibility, modular arithmetic, and number-theoretic functions. Number theory has ancient origins with contributions from mathematicians like Euclid, Fermat, Euler, and Gauss. Originally considered purely theoretical, modern number theory has found extensive applications in cryptography, computer science, and digital security. Key areas include prime number theory, modular arithmetic, congruence relations, Diophantine equations, and algebraic number theory. The field continues to address significant unsolved problems like the Riemann Hypothesis and Goldbach\'s Conjecture, while providing the mathematical foundation for secure digital communications.',
    'questions': number_theory_questions
}

logic_context = {
    'context': 'Logic is the systematic study of valid reasoning and inference. It provides formal tools to analyze arguments, distinguish valid from invalid reasoning, and establish the foundations of rational thought. Core areas include propositional logic, which studies truth-functional connectives; predicate logic, which adds quantifiers and predicates; modal logic, which addresses necessity and possibility; and non-classical logics like fuzzy logic and many-valued logic. Logic serves as the foundation for mathematics, computer science, and philosophical inquiry, with applications ranging from digital circuit design to artificial intelligence and natural language processing.',
    'questions': logic_questions
}

# Update the train_data list
new_train_data = [gdpr_section]
new_train_data.insert(1, sdn_context)
new_train_data.insert(2, number_theory_context)
new_train_data.insert(3, logic_context)
# Add existing domains
for i in range(1, len(data['train_data'])):
    new_train_data.append(data['train_data'][i])

data['train_data'] = new_train_data

# Save the updated dataset
with open('data/qa_en_dataset_fixed.json', 'w') as f:
    json.dump(data, f, indent=2)

print('Dataset structure fixed successfully. New file: data/qa_en_dataset_fixed.json') 