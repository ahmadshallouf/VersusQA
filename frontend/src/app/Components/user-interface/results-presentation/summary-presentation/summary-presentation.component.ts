import { Component } from '@angular/core';
import {AppStateService} from "../../../../StateManagement/app-state.service";

@Component({
  selector: 'app-summary-presentation',
  templateUrl: './summary-presentation.component.html',
  styleUrls: ['./summary-presentation.component.css']
})
export class SummaryPresentationComponent {

      constructor(
          public state: AppStateService
      ) {
      }
}
