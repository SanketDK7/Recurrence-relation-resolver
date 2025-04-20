import re
import math
import numpy as np
from sympy import symbols, Eq, solve, log, exp, sympify, lambdify, Poly
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
from PIL import Image as PILImage
from datetime import datetime
from flask import Flask, request, jsonify, send_file
import base64
import os

app = Flask(__name__)

def clean_extracted_text(text):
    text = text.replace(' ', '')
    text = re.sub(r'([a-zA-Z])\?', r'\1^2', text)
    clean_text = re.sub(r'[^0-9a-zA-Z\+\-\*/\^\(\)=xTnlog\s]', '', text)
    return clean_text

def plot_equation(equation_str, solution=None):
    try:
        x = symbols('x')
        if '=' in equation_str:
            lhs, rhs = equation_str.split('=')
            expr = sympify(lhs) - sympify(rhs)
        else:
            expr = sympify(equation_str)
            
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        
        f = lambdify(x, expr, 'numpy')
        x_vals = np.linspace(-10, 10, 400)
        y_vals = f(x_vals)
        
        ax.plot(x_vals, y_vals, label='y = ' + str(expr))
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True)
        
        if solution is not None:
            if isinstance(solution, (list, tuple)):
                for sol in solution:
                    try:
                        if hasattr(sol, 'is_real'):
                            if sol.is_real:
                                ax.plot(float(sol), 0, 'ro', label='Root at x=' + str(float(sol)))
                        else:
                            ax.plot(sol, 0, 'ro', label='Root at x=' + str(sol))
                    except (TypeError, ValueError):
                        continue
            else:
                try:
                    if hasattr(solution, 'is_real'):
                        if solution.is_real:
                            ax.plot(float(solution), 0, 'ro', label='Root at x=' + str(float(solution)))
                    else:
                        ax.plot(solution, 0, 'ro', label='Root at x=' + str(solution))
                except (TypeError, ValueError):
                    pass
        
        ax.legend()
        ax.set_title('Plot of ' + str(expr) + ' = 0')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return img_str
    except Exception as e:
        print('Plotting error: ' + str(e))
        return None

def plot_linear_system(equations, solution):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    
    for i, eq in enumerate(equations, 1):
        parts = eq.split(':')[1].strip().split('=')
        expr = parts[0].strip()
        a = float(expr.split('x')[0]) if 'x' in expr else 0
        b = float(expr.split('y')[0].split('+')[-1]) if 'y' in expr else 0
        c = float(parts[1].strip())
        
        if b != 0:
            y_vals = (c - a*x)/b
            ax.plot(x, y_vals, label='Eq ' + str(i) + ': ' + str(a) + 'x + ' + str(b) + 'y = ' + str(c))
        else:
            ax.axvline(x=c/a, label='Eq ' + str(i) + ': x = ' + str(c/a))
    
    if solution is not None:
        ax.plot(solution[0], solution[1], 'ro', label='Solution (' + str(solution[0]) + ', ' + str(solution[1]) + ')')
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True)
    ax.legend()
    ax.set_title('System of Linear Equations')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

"""def save_to_pdf(equation, solution, explanation="", plot_image=None):
    try:
        file_path = "solution.pdf"
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Customize styles with blue and pastel blue colors
        styles['Title'].textColor = colors.HexColor('#1E3A8A')  # Dark blue for title
        styles['Heading2'].textColor = colors.HexColor('#1E3A8A')  # Dark blue for headings
        styles['Normal'].backColor = colors.HexColor('#F0F9FF')  # Pastel blue background for text
        styles['Normal'].textColor = colors.HexColor('#1E3A8A')  # Dark blue for text

        story = []

        # Title and Date
        story.append(Paragraph("Equation Solution Report", styles['Title']))
        story.append(Paragraph("Generated on " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), styles['Normal']))
        story.append(Spacer(1, 24))

        # Equation
        story.append(Paragraph("Equation:", styles['Heading2']))
        story.append(Paragraph(equation, styles['Normal']))
        story.append(Spacer(1, 12))

        # Explanation
        if explanation:
            story.append(Paragraph("Explanation:", styles['Heading2']))
            story.append(Paragraph(explanation, styles['Normal']))
            story.append(Spacer(1, 12))

        # Solution Steps
        story.append(Paragraph("Solution Steps:", styles['Heading2']))
        for line in solution.split('\n'):
            if line.strip():
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))

        # Plot
        if plot_image:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Graphical Representation:", styles['Heading2']))
            buf = io.BytesIO(base64.b64decode(plot_image))
            img = PILImage.open(buf)
            temp_img_path = "temp_plot.png"
            img.save(temp_img_path, format="PNG")
            story.append(RLImage(temp_img_path, width=400, height=300))
            os.remove(temp_img_path)  # Clean up temporary file
            buf.close()

        doc.build(story)
        return file_path
    except Exception as e:
        raise Exception("Failed to save PDF: " + str(e))"""

def save_to_pdf(equation, solution, explanation="", plot_image=None):
    try:
        # Create a temporary directory for PDF generation
        temp_dir = os.path.join(os.getcwd(), "temp_pdf")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filenames
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        pdf_path = os.path.join(temp_dir, f"solution_{timestamp}.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Customize styles
        styles['Title'].textColor = colors.HexColor('#1E3A8A')
        styles['Heading2'].textColor = colors.HexColor('#1E3A8A')
        styles['Normal'].textColor = colors.HexColor('#1E3A8A')
        
        story = []
        
        # Title and Date
        story.append(Paragraph("Equation Solution Report", styles['Title']))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 24))
        
        # Equation
        story.append(Paragraph("Equation:", styles['Heading2']))
        story.append(Paragraph(equation, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Explanation
        if explanation:
            story.append(Paragraph("Explanation:", styles['Heading2']))
            story.append(Paragraph(explanation, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Solution Steps
        story.append(Paragraph("Solution Steps:", styles['Heading2']))
        if isinstance(solution, str):
            for line in solution.split('\n'):
                if line.strip():
                    story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(str(solution), styles['Normal']))
            story.append(Spacer(1, 6))
        
        # Plot handling with proper resource management
        if plot_image:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Graphical Representation:", styles['Heading2']))
            
            try:
                # Create temporary plot file with unique name
                plot_path = os.path.join(temp_dir, f"plot_{timestamp}.png")
                
                # Handle base64 encoded image
                if isinstance(plot_image, str):
                    img_data = base64.b64decode(plot_image)
                    with open(plot_path, 'wb') as f:
                        f.write(img_data)
                
                # Add image to PDF
                if os.path.exists(plot_path):
                    story.append(RLImage(plot_path, width=400, height=300))
                
            except Exception as e:
                story.append(Paragraph(f"Could not generate plot: {str(e)}", styles['Normal']))
        
        # Build the PDF document
        doc.build(story)
        
        # Read the generated PDF into memory
        with open(pdf_path, 'rb') as f:
            pdf_data = io.BytesIO(f.read())
        
        return pdf_data, temp_dir
        
    except Exception as e:
        raise Exception(f"Failed to save PDF: {str(e)}")

def solve_linear_equation(equation_text):
    import re
    import numpy as np
    
    equations = equation_text.split(';')
    steps = ["Solving system of linear equations:"]
    explanation = [
        "This is a system of linear equations in two variables (x and y).",
        "The solution is found by converting the equations to matrix form AX = B,",
        "where A is the coefficient matrix, X is the variable matrix, and B is the constant matrix.",
        "The solution is obtained by calculating X = A⁻¹B (matrix inversion method)."
    ]
    
    try:
        coefficients = []
        constants = []
        
        for i, eq in enumerate(equations, 1):
            eq = eq.replace(' ', '')
            # Updated regex pattern to handle more cases
            match = re.match(r'([+-]?\d*)?x([+-]?\d*)?y=([+-]?\d+)', eq)

            if match:
                # Handle coefficient of x
                a_str = match.group(1)
                if a_str in ["", "+"]:
                    a = 1
                elif a_str == "-":
                    a = -1
                else:
                    a = int(a_str)

                # Handle coefficient of y
                b_str = match.group(2)
                if b_str in ["", "+"]:
                    b = 1
                elif b_str == "-":
                    b = -1
                else:
                    b = int(b_str)

                # Handle constant term
                c = int(match.group(3))

                coefficients.append([a, b])
                constants.append(c)
                steps.append(f"Equation {i}: {a}x + {b}y = {c}")
            else:
                raise ValueError("Invalid equation format. Expected format: ax + by = c")
        
        A = np.array(coefficients)
        B = np.array(constants)
        
        steps.append("\nStep 1: Create matrices")
        steps.append("Coefficient matrix (A):\n" + str(A))
        steps.append("Constant matrix (B):\n" + str(B))
        
        steps.append("\nStep 2: Solve using matrix method")
        steps.append("We solve the equation AX = B, where X = [x y]")
        steps.append("Solution is X = A⁻¹B")
        
        try:
            A_inv = np.linalg.inv(A)
            steps.append("Inverse of A:\n" + str(A_inv))
            solution = np.linalg.solve(A, B)
            steps.append(f"\nSolution: x = {solution[0]:.6f}, y = {solution[1]:.6f}")

            plot_steps = []
            for i, (a, b, c) in enumerate(zip([row[0] for row in coefficients],
                                            [row[1] for row in coefficients],
                                            constants), 1):
                plot_steps.append(f"Equation {i}: {a}x + {b}y = {c}")

            return "\n".join(steps), solution, plot_steps, "\n".join(explanation)

        except np.linalg.LinAlgError:
            steps.append("\nSystem is either singular or not square.")
            steps.append("The equations may be dependent or inconsistent.")
            return "\n".join(steps), None, None, "\n".join(explanation)
    
    except Exception as e:
        return f"Error: {str(e)}", None, None, ""

def solve_quadratic_equation(equation):
    steps = ["Solving quadratic equation:"]
    explanation = [
        "This is a quadratic equation of the form ax² + bx + c = 0.",
        "The solutions are found using the quadratic formula:",
        "x = [-b ± √(b² - 4ac)] / (2a)",
        "The discriminant (D = b² - 4ac) determines the nature of the roots:",
        "- D > 0: Two distinct real roots",
        "- D = 0: One real root (repeated)",
        "- D < 0: Two complex roots"
    ]
    
    try:
        pattern = r'([+-]?\d*)x\^2\s*([+-]?\d*)x\s*([+-]?\d*)\s*=\s*([+-]?\d+)'
        match = re.match(pattern, equation.replace(' ', ''))
        
        if not match:
            raise ValueError("Invalid quadratic equation format. Expected format: ax^2 + bx + c = 0")
        
        a = float(match.group(1)) if match.group(1) not in ["", "+", "-"] else float(match.group(1)+"1")
        b = float(match.group(2)) if match.group(2) not in ["", "+", "-"] else float(match.group(2)+"1")
        c = float(match.group(3)) if match.group(3) not in ["", "+", "-"] else float(match.group(3)+"1")
        d = float(match.group(4))
        
        c = c - d
        
        steps.append("Standard form: " + str(a) + "x² + " + str(b) + "x + " + str(c) + " = 0")
        
        discriminant = b**2 - 4 * a * c
        steps.append("\nStep 1: Calculate discriminant")
        steps.append("D = b² - 4ac = " + str(b) + "² - 4*" + str(a) + "*" + str(c) + " = " + str(discriminant))

        if discriminant < 0:
            steps.append("\nStep 2: Analyze discriminant")
            steps.append("D < 0 → No real roots (complex roots exist)")
            sqrt_disc = math.sqrt(-discriminant)
            real_part = -b / (2 * a)
            imag_part = sqrt_disc / (2 * a)
            steps.append("Complex roots: " + str(real_part) + " ± " + str(imag_part) + "i")
            return "\n".join(steps), None, None, "\n".join(explanation)
        
        sqrt_disc = math.sqrt(discriminant)
        steps.append("√D = " + str(sqrt_disc))
        
        steps.append("\nStep 2: Apply quadratic formula")
        steps.append("x = [-b ± √(b² - 4ac)] / (2a)")
        
        x1 = (-b + sqrt_disc) / (2 * a)
        x2 = (-b - sqrt_disc) / (2 * a)
        
        steps.append("\nStep 3: Calculate roots")
        steps.append("x₁ = (-" + str(b) + " + " + str(sqrt_disc) + ") / (2*" + str(a) + ") = " + str(x1))
        steps.append("x₂ = (-" + str(b) + " - " + str(sqrt_disc) + ") / (2*" + str(a) + ") = " + str(x2))
        
        steps.append("\nSolution: x₁ = " + str(x1) + ", x₂ = " + str(x2))
        
        plot_expr = str(a) + "*x**2 + " + str(b) + "*x + " + str(c)
        return "\n".join(steps), [x1, x2], plot_expr, "\n".join(explanation)
    
    except Exception as e:
        return "Error: " + str(e), None, None, ""

def solve_recurrence_relation(equation):
    equation = equation.lower().replace(" ", "")
    steps = ["Solving recurrence relation:"]
    explanation = [
        "Recurrence relations describe functions in terms of their values at smaller inputs.",
        "The solution method depends on the form of the recurrence:",
        "1. Divide-and-conquer: T(n) = aT(n/b) + f(n) (use Master Theorem)",
        "2. Complex recurrence: T(n) = T(n-1) + T(n-2) + T(n-3) + ... + f(n)",
        "3. Variable coefficients: T(n) = nT(n-1) + f(n)"
    ]

    # Patterns for different types of recurrences
    divide_conquer_pattern = r't\(n\)=(\d+)t\(n/(\d+)\)\+?(.*)'
    complex_pattern = r't\(n\)=(t\(n-\d+\)(?:\+t\(n-\d+\))*)\+?(.*)'
    variable_coefficient_pattern = r't\(n\)=(\d*\.?\d*)\*?n\*?t\(n-(\d+)\)\+?(.*)'

    try:
        # Check for divide-and-conquer pattern
        match = re.match(divide_conquer_pattern, equation)
        if match:
            a = int(match.group(1))
            b = int(match.group(2))
            f_n = match.group(3).strip()
            steps.append(f"Recurrence relation: T(n) = {a}T(n/{b}) + {f_n}")
            explanation.append("\nThis is a divide-and-conquer recurrence relation where:")
            explanation.append(f"- a = {a} (number of subproblems)")
            explanation.append(f"- b = {b} (factor by which problem size is reduced)")
            explanation.append(f"- f(n) = {f_n} (cost of dividing/combining)")

            log_b_a = math.log(a, b)
            steps.append("\nStep 1: Calculate log_b(a)")
            steps.append(f"log_{b}({a}) = {log_b_a}")

            if 'n^' in f_n:
                degree = int(re.search(r'n\^(\d+)', f_n).group(1))
                steps.append(f"\nStep 2: Analyze f(n) = n^{degree}")
                if log_b_a > degree:
                    steps.append(f"Since log_b(a) > {degree}, Case 1 of Master Theorem applies")
                    steps.append(f"Time Complexity: O(n^{log_b_a})")
                elif log_b_a == degree:
                    steps.append(f"Since log_b(a) = {degree}, Case 2 of Master Theorem applies")
                    steps.append(f"Time Complexity: O(n^{degree} log n)")
                else:
                    steps.append(f"Since log_b_a < {degree}, Case 3 of Master Theorem applies")
                    steps.append(f"Time Complexity: O(n^{degree})")
            elif 'n' in f_n:
                steps.append("\nStep 2: Analyze f(n) = n")
                if log_b_a > 1:
                    steps.append("Since log_b(a) > 1, Case 1 of Master Theorem applies")
                    steps.append(f"Time Complexity: O(n^{log_b_a})")
                elif log_b_a == 1:
                    steps.append("Since log_b(a) = 1, Case 2 of Master Theorem applies")
                    steps.append("Time Complexity: O(n log n)")
                else:
                    steps.append("Since log_b(a) < 1, Case 3 of Master Theorem applies")
                    steps.append("Time Complexity: O(n)")
            else:
                steps.append("\nStep 2: Constant f(n)")
                steps.append(f"Time Complexity: O(n^{log_b_a})")
            
            steps.append("\nSpace Complexity: O(log n) [due to recursion depth]")
            return "\n".join(steps), None, None, "\n".join(explanation)

        # Check for complex recurrence pattern
        match = re.match(complex_pattern, equation)
        if match:
            terms = re.findall(r't\(n-(\d+)\)', match.group(1))
            f_n = match.group(2).strip() if match.group(2) else ""
            
            # Convert terms to integers
            terms = [int(k) for k in terms]
            
            steps.append("Complex recurrence relation detected:")
            terms_str = " + ".join(f"T(n-{k})" for k in terms)
            steps.append(f"T(n) = {terms_str}{' + ' + f_n if f_n else ''}")
            
            explanation.append("\nThis is a complex recurrence relation where:")
            explanation.append(f"- The function calls itself with {len(terms)} different terms")
            explanation.append(f"- Terms: {', '.join(f'n-{k}' for k in terms)}")
            if f_n:
                explanation.append(f"- Additional work per step: {f_n}")

            steps.append("\nStep 1: Analyze recurrence structure")
            if len(terms) == 2 and terms == [1, 2]:
                steps.append("This is a Fibonacci-like recurrence relation")
                steps.append("Time Complexity: O(φⁿ) where φ ≈ 1.618 (golden ratio)")
            elif len(terms) == 3 and terms == [1, 2, 3]:
                steps.append("This is a Tribonacci-like recurrence relation")
                steps.append("Time Complexity: O(αⁿ) where α ≈ 1.8393")
                steps.append("Note: α is the real root of x³ - x² - x - 1 = 0")
            else:
                max_term = max(terms)
                steps.append(f"This is a {len(terms)}-term recurrence relation")
                growth_rate = len(terms) ** (1/max_term)
                steps.append(f"Time Complexity: O({growth_rate:.4f}ⁿ)")
                steps.append("Note: This is an approximation based on the number and size of terms")

            max_depth = max(terms)
            steps.append(f"\nSpace Complexity: O(n) [due to recursion depth of {max_depth}]")
            
            if f_n:
                steps.append("\nAdditional Analysis:")
                steps.append(f"The recurrence includes a non-homogeneous term: {f_n}")
                steps.append("This may affect the actual complexity by a polynomial factor")

            return "\n".join(steps), None, None, "\n".join(explanation)

        # Check for variable coefficient pattern
        match = re.match(variable_coefficient_pattern, equation)
        if match:
            coefficient = float(match.group(1)) if match.group(1) and match.group(1) not in ['', '+'] else 1
            k = int(match.group(2))
            f_n = match.group(3).strip()
            steps.append(f"Recurrence relation: T(n) = {coefficient}nT(n-{k}) + {f_n}")
            explanation.append("\nThis is a variable coefficient recurrence relation where:")
            explanation.append(f"- The coefficient depends on n: {coefficient}n")
            explanation.append(f"- The function calls itself with smaller input n-{k}")
            explanation.append(f"- Additional work per step: {f_n}")

            steps.append("\nStep 1: Analyze recurrence type")
            if coefficient == 1 and k == 1:
                steps.append("This is a factorial-like recurrence relation")
                steps.append("Time Complexity: O(n!)")
            else:
                steps.append("This is a variable coefficient recurrence")
                steps.append("Time Complexity: O(n!)")
            
            steps.append("\nSpace Complexity: O(n) [due to recursion depth]")
            return "\n".join(steps), None, None, "\n".join(explanation)

        raise ValueError("Unrecognized recurrence relation format. Supported formats:\n"
                        "1. Divide-and-conquer: T(n) = aT(n/b) + f(n)\n"
                        "2. Complex recurrence: T(n) = T(n-1) + T(n-2) + T(n-3) + ... + f(n)\n"
                        "3. Variable coefficients: T(n) = nT(n-1) + f(n)")

    except Exception as e:
        return f"Error: {str(e)}", None, None, ""

def solve_polynomial_equation(equation):
    steps = ["Solving polynomial equation:"]
    explanation = [
        "This is a polynomial equation where we need to find the roots (solutions).",
        "The degree of the polynomial determines the number of possible roots:",
        "- Degree 1 (linear): 1 root",
        "- Degree 2 (quadratic): 2 roots",
        "- Degree 3 (cubic): 3 roots",
        "- And so on...",
        "The solutions may be real or complex numbers."
    ]
    
    try:
        # Clean up the equation
        equation = equation.replace(' ', '').replace('³', '**3').replace('²', '**2')
        
        # Split equation into left and right sides
        if '=' not in equation:
            raise ValueError("Equation must contain '='")
        
        lhs, rhs = equation.split('=')
        
        # Move everything to the left side
        expr = f"({lhs})-({rhs})"
        
        x = symbols('x')
        expr = sympify(expr)
        
        steps.append(f"Original equation: {equation}")
        steps.append("\nStep 1: Rearrange to standard form")
        steps.append(f"{expr} = 0")
        
        # Get the degree of the polynomial
        degree = Poly(expr, x).degree()
        steps.append(f"\nStep 2: Identify degree of polynomial")
        steps.append(f"This is a degree {degree} polynomial")
        
        # Solve the equation
        solution = solve(expr, x)
        steps.append("\nStep 3: Find roots")
        
        if not solution:
            steps.append("No solutions found")
        else:
            for i, sol in enumerate(solution, 1):
                # Convert complex solutions to a nicer format
                if sol.is_real:
                    steps.append(f"Root {i}: x = {float(sol)}")
                else:
                    real_part = re.sub(r'\.0$', '', str(float(re.search(r'-?\d+\.?\d*', str(sol.real)).group())))
                    imag_part = re.sub(r'\.0$', '', str(float(re.search(r'-?\d+\.?\d*', str(sol.imag)).group())))
                    steps.append(f"Root {i}: x = {real_part} + {imag_part}i")
        
        # Create plot expression
        plot_expr = str(expr)
        return "\n".join(steps), solution, plot_expr, "\n".join(explanation)
    
    except Exception as e:
        return f"Error: {str(e)}", None, None, ""

def solve_exponential_logarithmic_equation(equation):
    steps = ["Solving exponential/logarithmic equation:"]
    explanation = [
        "This equation involves exponential or logarithmic functions.",
        "Solution methods depend on the specific form:",
        "- Exponential equations: Use logarithms to bring exponents down",
        "- Logarithmic equations: Convert to exponential form",
        "The solution may require algebraic manipulation before solving."
    ]
    
    try:
        # Clean up the equation
        equation = (equation.replace(' ', '')
                   .replace('ˣ', '**x')
                   .replace('ln', 'log')
                   .replace('e^', 'exp'))
        
        if '=' not in equation:
            raise ValueError("Equation must contain '='")
        
        x = symbols('x')
        lhs, rhs = equation.split('=')
        
        # Convert the equation to symbolic form
        lhs_expr = sympify(lhs)
        rhs_expr = sympify(rhs)
        eq = Eq(lhs_expr, rhs_expr)
        
        steps.append(f"Original equation: {equation}")
        steps.append("\nStep 1: Convert to standard form")
        steps.append(f"{lhs_expr} = {rhs_expr}")
        
        # Solve the equation
        solution = solve(eq, x)
        steps.append("\nStep 2: Solve for x")
        
        if not solution:
            steps.append("No solutions found")
        else:
            for i, sol in enumerate(solution, 1):
                if sol.is_real:
                    steps.append(f"Solution {i}: x = {float(sol)}")
                else:
                    steps.append(f"Solution {i}: x = {sol}")
        
        # Create plot expression
        plot_expr = f"({lhs}) - ({rhs})"
        return "\n".join(steps), solution, plot_expr, "\n".join(explanation)
    
    except Exception as e:
        return f"Error: {str(e)}", None, None, ""

def solve_inequality(inequality):
    steps = ["Solving inequality:"]
    explanation = [
        "This is an inequality involving mathematical expressions.",
        "Solution methods:",
        "- Convert to standard form (all terms on one side)",
        "- Find critical points by solving corresponding equation",
        "- Test intervals between critical points",
        "- Express solution in interval notation"
    ]
    
    try:
        # Clean up the inequality
        inequality = (inequality.replace(' ', '')
                     .replace('≤', '<=')
                     .replace('≥', '>=')
                     .replace('²', '**2'))
        
        # Extract operator and convert to standard form
        op = re.findall(r'[<>]=?', inequality)[0]
        lhs, rhs = re.split(r'[<>]=?', inequality)
        
        x = symbols('x')
        expr = sympify(f"({lhs})-({rhs})")
        
        steps.append(f"Original inequality: {inequality}")
        steps.append("\nStep 1: Rearrange to standard form")
        steps.append(f"{expr} {op} 0")
        
        # Find critical points
        critical_points = solve(expr, x)
        steps.append("\nStep 2: Find critical points")
        if critical_points:
            for i, point in enumerate(critical_points, 1):
                steps.append(f"Critical point {i}: x = {point}")
        else:
            steps.append("No critical points found")
        
        # Determine intervals
        if critical_points:
            steps.append("\nStep 3: Test intervals")
            # Sort critical points
            points = sorted([float(p) for p in critical_points if p.is_real])
            
            # Test points in each interval
            intervals = []
            test_points = [-1000] + points + [1000]
            for i in range(len(test_points) - 1):
                test_x = (test_points[i] + test_points[i+1]) / 2
                result = expr.subs(x, test_x)
                if eval(f"result {op} 0"):
                    if i == 0:
                        intervals.append(f"x < {points[0]}")
                    elif i == len(test_points) - 2:
                        intervals.append(f"x > {points[-1]}")
                    else:
                        intervals.append(f"{points[i]} < x < {points[i+1]}")
            
            solution = " or ".join(intervals)
            steps.append(f"\nSolution: {solution}")
        
        # Create plot expression
        plot_expr = str(expr)
        return "\n".join(steps), solution, plot_expr, "\n".join(explanation)
    
    except Exception as e:
        return f"Error: {str(e)}", None, None, ""

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/solve_math', methods=['POST'])
def solve_math():
    data = request.get_json()
    equation = data['equation']
    
    try:
        # Clean up input
        equation = equation.strip()
        
        # Determine equation type and solve
        if ';' in equation and 'x' in equation and 'y' in equation:
            # Linear system
            result, solution, plot_info, explanation = solve_linear_equation(equation)
            plot_image = plot_linear_system(plot_info, solution) if solution is not None else None
            
        elif 'T(n)' in equation or 't(n)' in equation:
            # Recurrence relation
            result, solution, _, explanation = solve_recurrence_relation(equation)
            plot_image = None  # No plotting for recurrence relations
            
        elif any(op in equation for op in ['<', '>', '≤', '≥']):
            # Inequality
            result, solution, plot_expr, explanation = solve_inequality(equation)
            plot_image = plot_equation(plot_expr, solution) if solution is not None else None
            
        elif 'log' in equation.lower() or 'ln' in equation.lower() or 'ˣ' in equation or 'exp' in equation:
            # Exponential/logarithmic
            result, solution, plot_expr, explanation = solve_exponential_logarithmic_equation(equation)
            plot_image = plot_equation(plot_expr, solution) if solution is not None else None
            
        elif 'x^3' in equation or 'x³' in equation:
            # Cubic or higher degree polynomial
            result, solution, plot_expr, explanation = solve_polynomial_equation(equation)
            plot_image = plot_equation(plot_expr, solution) if solution is not None else None
            
        elif 'x^2' in equation or 'x²' in equation:
            # Quadratic equation
            result, solution, plot_expr, explanation = solve_quadratic_equation(equation)
            plot_image = plot_equation(plot_expr, solution) if solution is not None else None
            
        elif '^' in equation:
            # Higher degree polynomial
            result, solution, plot_expr, explanation = solve_polynomial_equation(equation)
            plot_image = plot_equation(plot_expr, solution) if solution is not None else None
            
        else:
            result = "Unsupported equation type"
            solution = None
            plot_image = None
            explanation = ""

        return jsonify({
            'result': result,
            'plot_image': plot_image,
            'plot_available': plot_image is not None,
            'explanation': explanation
        })
    
    except Exception as e:
        return jsonify({
            'result': f"Error: {str(e)}",
            'plot_image': None,
            'plot_available': False,
            'explanation': ""
        })

@app.route('/solve_recurrence', methods=['POST'])
def solve_recurrence():
    data = request.get_json()
    recurrence = data['recurrence']
    
    try:
        result, _, _, explanation = solve_recurrence_relation(recurrence)
        return jsonify({'result': result, 'explanation': explanation})
    except Exception as e:
        return jsonify({'result': "Error: " + str(e), 'explanation': ""})

# [Previous imports remain the same...]

def save_to_pdf(equation, solution, explanation="", plot_image=None):
    """Single, unified version of the PDF saving function"""
    try:
        file_path = "solution.pdf"
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Customize styles
        styles['Title'].textColor = colors.HexColor('#1E3A8A')
        styles['Heading2'].textColor = colors.HexColor('#1E3A8A')
        styles['Normal'].backColor = colors.HexColor('#F0F9FF')
        styles['Normal'].textColor = colors.HexColor('#1E3A8A')

        story = []
        
        # Title and Date
        story.append(Paragraph("Equation Solution Report", styles['Title']))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 24))
        
        # Equation
        story.append(Paragraph("Equation:", styles['Heading2']))
        story.append(Paragraph(equation, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Explanation
        if explanation:
            story.append(Paragraph("Explanation:", styles['Heading2']))
            story.append(Paragraph(explanation, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Solution Steps
        story.append(Paragraph("Solution Steps:", styles['Heading2']))
        if isinstance(solution, str):
            # Handle string solutions
            for line in solution.split('\n'):
                if line.strip():
                    story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 6))
        else:
            # Handle other solution formats
            story.append(Paragraph(str(solution), styles['Normal']))
            story.append(Spacer(1, 6))
        
        # Plot handling
        if plot_image:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Graphical Representation:", styles['Heading2']))
            
            try:
                if isinstance(plot_image, str):
                    # Base64 encoded image
                    buf = io.BytesIO(base64.b64decode(plot_image))
                    img = PILImage.open(buf)
                else:
                    # Matplotlib figure
                    buf = io.BytesIO()
                    plot_image.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    img = PILImage.open(buf)
                
                temp_img_path = "temp_plot.png"
                img.save(temp_img_path, format="PNG")
                
                if os.path.exists(temp_img_path):
                    story.append(RLImage(temp_img_path, width=400, height=300))
                    os.remove(temp_img_path)
                buf.close()
            except Exception as e:
                story.append(Paragraph(f"Could not generate plot: {str(e)}", styles['Normal']))
        
        doc.build(story)
        return file_path
    except Exception as e:
        raise Exception(f"Failed to save PDF: {str(e)}")

@app.route('/save_pdf', methods=['POST'])
def save_pdf():
    temp_dir = None
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        mode = data.get('mode', 'math')
        equation = data.get('input', '')
        solution = data.get('output', '')
        plot_image = data.get('plot_image')
        explanation = data.get('explanation', '')

        if not equation or not solution:
            return jsonify({'error': 'Equation and solution are required'}), 400

        # Generate PDF
        pdf_buffer, temp_dir = save_to_pdf(
            equation=equation,
            solution=solution,
            explanation=explanation,
            plot_image=plot_image
        )

        # Create response
        response = send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='solution.pdf',
            mimetype='application/pdf'
        )
        
        # Cleanup function
        def cleanup():
            try:
                if temp_dir and os.path.exists(temp_dir):
                    for filename in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print(f"Error deleting file {file_path}: {e}")
                    os.rmdir(temp_dir)
                pdf_buffer.close()
            except Exception as e:
                print(f"Cleanup error: {e}")
        
        response.call_on_close(cleanup)
        return response
        
    except Exception as e:
        if temp_dir and os.path.exists(temp_dir):
            try:
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                os.rmdir(temp_dir)
            except:
                pass
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)