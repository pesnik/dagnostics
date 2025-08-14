#!/usr/bin/env python3
"""
Simple web server to demonstrate the feedback interface
"""


from dagnostics.web.feedback import FEEDBACK_HTML_TEMPLATE


def create_sample_feedback_page():
    """Create a sample feedback page with mock data"""

    # Sample log context and analysis
    sample_log = (
        "[2025-08-10T16:13:23.140+0600] {taskinstance.py:1225} INFO - "
        "Marking task as FAILED. dag_id=downstream_bts_wise_service_revenue_monthly, "
        "task_id=tpt_bts_wise_service_revenue_monthly_export\n"
        "[2025-08-10T16:13:23.581+0600] {logging_mixin.py:190} INFO - "
        "Failure caused by TPT Error: TPT_INFRA: TPT04183: Error: Line 1 of Command Line:\n"
        "[2025-08-10T16:13:23.597+0600] {standard_task_runner.py:124} ERROR - "
        "Failed to execute job 15285469 for task tpt_bts_wise_service_revenue_monthly_export "
        "(TPT Error: TPT_INFRA: TPT04183: Error: Line 1 of Command Line:"
    )

    sample_analysis = {
        "error_message": "TPT Error: TPT_INFRA: TPT04183: Error: Line 1 of Command Line:",
        "category": "configuration_error",
        "severity": "high",
        "confidence": 0.85,
        "reasoning": "TPT export template configuration error on command line parsing",
    }

    # Use the HTML template directly and populate with sample data
    feedback_html = FEEDBACK_HTML_TEMPLATE

    # Add JavaScript to populate the form with sample data
    escaped_log = sample_log.replace("`", "\\`")
    sample_js = f"""
    <script>
        // Populate sample data when page loads
        window.onload = function() {{
            document.getElementById('log-context').innerHTML = `<pre>{escaped_log}</pre>`;
            document.getElementById('original-analysis').innerHTML = `
                <p><strong>Error:</strong> {sample_analysis['error_message']}</p>
                <p><strong>Category:</strong> {sample_analysis['category']}</p>
                <p><strong>Severity:</strong> {sample_analysis['severity']}</p>
                <p><strong>Confidence:</strong> {sample_analysis['confidence']}</p>
                <p><strong>Reasoning:</strong> {sample_analysis['reasoning']}</p>
            `;
        }};
    </script>
    """

    # Insert sample data script before closing body tag
    feedback_html = feedback_html.replace("</body>", sample_js + "</body>")

    return feedback_html


def main():
    """Start simple HTTP server with feedback interface"""
    import os
    import tempfile
    import webbrowser

    # Create sample feedback page
    feedback_html = create_sample_feedback_page()

    # Write to temporary HTML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(feedback_html)
        temp_file = f.name

    print(f"📝 Sample feedback page created: {temp_file}")
    print("\n🌐 Opening feedback interface in browser...")
    print("   This shows how the feedback interface looks with sample data")
    print(
        "\n💡 In production, this would be integrated with your DAGnostics analysis workflow"
    )

    # Open in browser
    webbrowser.open(f"file://{temp_file}")

    print("\n✅ Feedback interface opened!")
    print("\nFeatures demonstrated:")
    print("• Log context display")
    print("• Original AI analysis review")
    print("• Correction form with categories")
    print("• Rating system (1-5 stars)")
    print("• Additional comments")

    print(f"\n📁 HTML file: {temp_file}")
    print("\nPress Ctrl+C to clean up...")

    try:
        input("Press Enter to clean up and exit...")
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        os.unlink(temp_file)
        print("🧹 Cleaned up temporary file")


if __name__ == "__main__":
    main()
