---
name: collect-doc
description: Collect information on the web for a particular topic
license: MIT
metadata: 
  tested_on: chat-gpt-5.3-medium
  author: "Giovanni Giorgi <jj@gioorgi.com>"
  version: "0.1"
compatibility: Codex, Claude Code, Copilot and products.
---

## When to use this skill

When user explicit ask for it

## Guidelines

Name the requested information wih a two-word identifier we will call $id

- Create a folder under doc/$id/
- Download all the relevant information from the Internet.
- Put all the information you find under under doc/$id
- Create a file named doc/$id/reference.md with the list of all the url scanned
- Create a summary of the information inside doc/$id/README.md

## How to organize the information

Organize the information in numbered folders, from 10 to 90.
Example structure

- 10-intro
- 20-wikipedia-reference
- 30-sub-topic1
- 40-sub-topic2
- ...
- 90-conclusion
  
For every subtopic you can further organize the documentation

