import pandas as pd
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from typing import List, Dict

def export_to_csv(price_data: pd.Series, sentiment_data: pd.Series,
                 metrics: Dict, insights: List[Dict], tweets: List[Dict]) -> str:
    """Export data to CSV format."""
    # Create a buffer to store the CSV data
    buffer = io.StringIO()
    
    # Write header
    buffer.write("Crypto Sentiment Analysis Report\n")
    buffer.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Write metrics
    buffer.write("Metrics\n")
    buffer.write("-------\n")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            buffer.write(f"{key}: {value:.4f}\n")
        else:
            buffer.write(f"{key}: {value}\n")
    buffer.write("\n")
    
    # Write insights
    buffer.write("Insights\n")
    buffer.write("--------\n")
    for insight in insights:
        buffer.write(f"- {insight}\n")
    buffer.write("\n")
    
    # Write price and sentiment data
    buffer.write("Date,Price,Sentiment\n")
    for date in price_data.index:
        buffer.write(f"{date.strftime('%Y-%m-%d')},{price_data[date]:.2f},{sentiment_data[date]:.4f}\n")
    buffer.write("\n")
    
    # Write tweets
    buffer.write("Tweets\n")
    buffer.write("------\n")
    for tweet in tweets:
        buffer.write(f"Date: {tweet['date']}\n")
        buffer.write(f"Text: {tweet['text']}\n")
        buffer.write(f"Sentiment: {tweet['sentiment']:.4f}\n\n")
    
    return buffer.getvalue()

def export_to_pdf(price_data: pd.Series, sentiment_data: pd.Series,
                 metrics: Dict, insights: List[Dict], tweets: List[Dict],
                 selected_crypto: str) -> bytes:
    """Export data to PDF format."""
    # Create a buffer to store the PDF data
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    elements.append(Paragraph(f"{selected_crypto} Sentiment Analysis Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Add metrics
    elements.append(Paragraph("Metrics", styles['Heading2']))
    metrics_data = [[k, str(v)] for k, v in metrics.items()]
    metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 20))
    
    # Add insights
    elements.append(Paragraph("Insights", styles['Heading2']))
    for insight in insights:
        elements.append(Paragraph(f"• {insight}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Add price and sentiment data
    elements.append(Paragraph("Price and Sentiment Data", styles['Heading2']))
    data = [['Date', 'Price', 'Sentiment']]
    for date in price_data.index:
        data.append([
            date.strftime('%Y-%m-%d'),
            f"{price_data[date]:.2f}",
            f"{sentiment_data[date]:.4f}"
        ])
    data_table = Table(data, colWidths=[2*inch, 2*inch, 2*inch])
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(data_table)
    elements.append(Spacer(1, 20))
    
    # Add tweets
    elements.append(Paragraph("Sample Tweets", styles['Heading2']))
    for tweet in tweets:
        elements.append(Paragraph(f"Date: {tweet['date']}", styles['Normal']))
        elements.append(Paragraph(f"Text: {tweet['text']}", styles['Normal']))
        elements.append(Paragraph(f"Sentiment: {tweet['sentiment']:.4f}", styles['Normal']))
        elements.append(Spacer(1, 10))
    
    # Build the PDF
    doc.build(elements)
    
    # Get the PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data 