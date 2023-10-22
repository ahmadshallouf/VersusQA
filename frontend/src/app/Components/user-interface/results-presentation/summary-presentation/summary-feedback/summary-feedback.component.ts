import { Component } from '@angular/core';

@Component({
  selector: 'app-summary-feedback',
  templateUrl: './summary-feedback.component.html',
  styleUrls: ['./summary-feedback.component.css']
})
export class SummaryFeedbackComponent {

    useful = false;
    fluent = false;

    constructor(
    )
    { }

    submitFeedback(): void {

    }

}
